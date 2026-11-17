"""Embedder interface shared by every backend."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into fixed-size, L2-normalised vectors."""

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, one vector per input, in order."""
        ...


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Embed a single text."""
    return embedder.embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors of equal length."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
