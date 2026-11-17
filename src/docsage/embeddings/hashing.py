"""Deterministic bag-of-words embedder based on the hashing trick.

It needs no model and no network, so tests and offline demos get stable vectors.
Texts that share words get a high cosine similarity, which is enough to exercise
retrieval code paths.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    def __init__(self, dimension: int = 256) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self._dimension] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
