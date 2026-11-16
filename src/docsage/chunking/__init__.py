"""Chunking strategies."""

from __future__ import annotations

from docsage.chunking.base import Chunk, Chunker
from docsage.chunking.fixed import FixedSizeChunker
from docsage.chunking.recursive import RecursiveChunker
from docsage.chunking.sentences import SentenceWindowChunker
from docsage.loaders.base import Document


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk several documents; chunk indexes restart for each document."""
    return [c for doc in docs for c in chunker.chunk(doc)]


__all__ = [
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "chunk_documents",
]
