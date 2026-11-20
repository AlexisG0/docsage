"""Reranking: rescoring fused candidates with a model that reads query and passage together."""

from __future__ import annotations

import re
from typing import Any, Protocol

from docsage.vectorstore.base import SearchHit


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def rerank(
    reranker: Reranker, query: str, hits: list[SearchHit], k: int | None = None
) -> list[SearchHit]:
    """Reorder ``hits`` by reranker score. Ties keep the incoming order. Returns the top ``k``."""
    if not hits:
        return []
    scores = reranker.score(query, [h.chunk.text for h in hits])
    if len(scores) != len(hits):
        raise ValueError("reranker returned a different number of scores than passages")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    out = [hits[i].model_copy(update={"score": scores[i]}) for i in order]
    return out if k is None else out[: max(k, 0)]


_TOKEN = re.compile(r"\w+")


class OverlapReranker:
    """Deterministic fake: the fraction of distinct query terms present in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = set(_TOKEN.findall(query.lower()))
        if not terms:
            return [0.0] * len(passages)
        return [len(terms & set(_TOKEN.findall(p.lower()))) / len(terms) for p in passages]


class CrossEncoderReranker:
    """sentence-transformers cross-encoder. Install with ``pip install docsage[local]``."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; run `pip install docsage[local]`"
            ) from exc
        self._model: Any = CrossEncoder(model_name)
        self._batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        out = self._model.predict([(query, p) for p in passages], batch_size=self._batch_size)
        return [float(x) for x in out]
