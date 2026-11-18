"""Vector store interface shared by every backend."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from docsage.chunking.base import Chunk

# Fields a metadata filter may target.
FILTERABLE_FIELDS = ("path", "title", "page", "index")


class SearchHit(BaseModel):
    """A chunk returned by a search, with its similarity score (higher is better)."""

    id: str
    chunk: Chunk
    score: float = Field(description="Cosine similarity between query and chunk")


def chunk_id(chunk: Chunk) -> str:
    """Stable identifier of a chunk: source path, page and position in the document."""
    page = chunk.page if chunk.page is not None else 0
    return f"{chunk.path}#p{page}#{chunk.index}"


class VectorStore(Protocol):
    """Stores chunk vectors and answers nearest-neighbour queries."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Insert or replace chunks (matched by id). Returns the number written."""
        ...

    def delete(self, *, path: str | None = None, ids: list[str] | None = None) -> int:
        """Delete by source path and/or chunk ids. Returns the number removed."""
        ...

    def search(
        self,
        vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the ``k`` most similar chunks, optionally restricted by metadata equality."""
        ...

    def count(self) -> int:
        """Number of stored chunks."""
        ...
