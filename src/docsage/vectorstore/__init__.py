"""Vector stores.

``LanceVectorStore`` needs the optional ``lance`` extra and is imported lazily.
"""

from __future__ import annotations

from typing import Any

from docsage.vectorstore.base import SearchHit, VectorStore, chunk_id
from docsage.vectorstore.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "LanceVectorStore", "SearchHit", "VectorStore", "chunk_id"]


def __getattr__(name: str) -> Any:
    if name == "LanceVectorStore":
        from docsage.vectorstore.lance import LanceVectorStore

        return LanceVectorStore
    raise AttributeError(name)
