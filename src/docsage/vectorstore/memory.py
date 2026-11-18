"""In-memory vector store: the reference implementation, used in tests and small corpora."""

from __future__ import annotations

import math
from typing import Any

from docsage.chunking.base import Chunk
from docsage.vectorstore.base import FILTERABLE_FIELDS, SearchHit, chunk_id


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _field(chunk: Chunk, key: str) -> Any:
    # Mirror the on-disk backend: a missing title/page is "" / 0.
    if key == "title":
        return chunk.title or ""
    if key == "page":
        return chunk.page or 0
    return getattr(chunk, key)


class InMemoryVectorStore:
    """Keeps chunks and vectors in a dict. Nothing is persisted."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for vec in vectors:
            if len(vec) != self.dimension:
                raise ValueError(f"vector has {len(vec)} dims, expected {self.dimension}")
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk_id(chunk)] = (chunk, list(vec))
        return len(chunks)

    def delete(self, *, path: str | None = None, ids: list[str] | None = None) -> int:
        if path is None and ids is None:
            return 0
        doomed = [
            rid
            for rid, (chunk, _) in self._rows.items()
            if (path is None or chunk.path == path) and (ids is None or rid in ids)
        ]
        for rid in doomed:
            del self._rows[rid]
        return len(doomed)

    def search(
        self,
        vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        filters = filters or {}
        for key in filters:
            if key not in FILTERABLE_FIELDS:
                raise ValueError(
                    f"cannot filter on {key!r}; allowed: {', '.join(FILTERABLE_FIELDS)}"
                )
        if k <= 0:
            return []
        hits = [
            SearchHit(id=rid, chunk=chunk, score=_cosine(vector, vec))
            for rid, (chunk, vec) in self._rows.items()
            if all(_field(chunk, key) == value for key, value in filters.items())
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        return len(self._rows)
