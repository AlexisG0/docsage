"""Embedding providers."""

from __future__ import annotations

from typing import Any

from docsage.embeddings.base import Embedder, cosine, embed_query
from docsage.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "cosine", "embed_query", "load_embedder"]


def load_embedder(name: str = "hashing", **kwargs: Any) -> Embedder:
    """Build an embedder by name: ``hashing`` or ``sentence-transformers``."""
    if name == "hashing":
        return HashingEmbedder(**kwargs)
    if name == "sentence-transformers":
        from docsage.embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder {name!r}; choose 'hashing' or 'sentence-transformers'")
