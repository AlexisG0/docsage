"""Hybrid retrieval: BM25 and vector search fused with weighted reciprocal rank fusion."""

from __future__ import annotations

from typing import Any

from docsage.embeddings.base import Embedder, embed_query
from docsage.keyword.bm25 import BM25Index
from docsage.vectorstore.base import SearchHit, VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[SearchHit]:
    """Fuse ranked lists: each hit scores ``sum(weight / (k + rank))`` over the lists it is in.

    Ranks start at 1. Only ranks matter, so scores from different retrievers (BM25 vs cosine)
    need no normalisation. Ties are broken by id so the output is deterministic. The returned
    hits carry the fused score.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    scores: dict[str, float] = {}
    first: dict[str, SearchHit] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (k + rank)
            first.setdefault(hit.id, hit)
    fused = [first[i].model_copy(update={"score": s}) for i, s in scores.items()]
    fused.sort(key=lambda h: (-h.score, h.id))
    return fused


class HybridRetriever:
    """Runs keyword and vector search and combines them with reciprocal rank fusion.

    ``keyword_weight`` and ``vector_weight`` scale each retriever's contribution; a weight of
    0 skips that retriever. Each retriever fetches ``candidates`` hits (default ``4 * k``)
    before fusion so documents ranked low by one side can still be lifted by the other.
    """

    def __init__(
        self,
        store: VectorStore,
        index: BM25Index,
        embedder: Embedder,
        *,
        keyword_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        if keyword_weight < 0 or vector_weight < 0:
            raise ValueError("weights must be non-negative")
        if keyword_weight == 0 and vector_weight == 0:
            raise ValueError("at least one weight must be positive")
        self.store = store
        self.index = index
        self.embedder = embedder
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        candidates: int | None = None,
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        n = candidates if candidates is not None else 4 * k
        n = max(n, k)
        rankings: list[list[SearchHit]] = []
        weights: list[float] = []
        if self.keyword_weight > 0:
            rankings.append(self.index.search(query, k=n, filters=filters))
            weights.append(self.keyword_weight)
        if self.vector_weight > 0:
            vector = embed_query(self.embedder, query)
            rankings.append(self.store.search(vector, k=n, filters=filters))
            weights.append(self.vector_weight)
        return reciprocal_rank_fusion(rankings, weights, self.rrf_k)[:k]
