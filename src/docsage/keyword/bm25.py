"""BM25 keyword index with JSON persistence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from docsage.chunking.base import Chunk
from docsage.vectorstore.base import FILTERABLE_FIELDS, SearchHit, chunk_id

_TOKEN = re.compile(r"[a-z0-9_]+")
FORMAT_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (underscores kept, so ``max_chars`` stays one term)."""
    return _TOKEN.findall(text.lower())


def _field(chunk: Chunk, key: str) -> Any:
    if key == "title":
        return chunk.title or ""
    if key == "page":
        return chunk.page or 0
    return getattr(chunk, key)


class BM25Index:
    """Okapi BM25 over chunk text, keyed by the same ids as the vector stores.

    Scores are not comparable across indexes; use rank-based fusion to combine them
    with vector scores.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> int:
        """Index chunks, replacing any with the same id. Returns the number written."""
        for chunk in chunks:
            cid = chunk_id(chunk)
            self._remove(cid)
            tf = Counter(tokenize(chunk.text))
            self._chunks[cid] = chunk
            self._tf[cid] = tf
            self._df.update(tf.keys())
            self._total_len += sum(tf.values())
        return len(chunks)

    def _remove(self, cid: str) -> bool:
        tf = self._tf.pop(cid, None)
        if tf is None:
            return False
        del self._chunks[cid]
        for term in tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._total_len -= sum(tf.values())
        return True

    def delete(self, *, path: str | None = None, ids: list[str] | None = None) -> int:
        """Remove chunks by source path and/or id. Returns the number removed."""
        if path is None and ids is None:
            return 0
        wanted = set(ids) if ids is not None else None
        doomed = [
            cid
            for cid, chunk in self._chunks.items()
            if (path is None or chunk.path == path) and (wanted is None or cid in wanted)
        ]
        return sum(self._remove(cid) for cid in doomed)

    def search(
        self, query: str, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        filters = filters or {}
        for key in filters:
            if key not in FILTERABLE_FIELDS:
                raise ValueError(
                    f"cannot filter on {key!r}; allowed: {', '.join(FILTERABLE_FIELDS)}"
                )
        terms = set(tokenize(query))
        if k <= 0 or not self._chunks or not terms:
            return []
        n = len(self._chunks)
        avg_len = self._total_len / n or 1.0
        idf = {
            t: math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            for t in terms
            if t in self._df
        }
        hits: list[SearchHit] = []
        for cid, tf in self._tf.items():
            chunk = self._chunks[cid]
            if not all(_field(chunk, key) == value for key, value in filters.items()):
                continue
            norm = self.k1 * (1 - self.b + self.b * sum(tf.values()) / avg_len)
            score = sum(
                w * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t, w in idf.items() if t in tf
            )
            if score > 0:
                hits.append(SearchHit(id=cid, chunk=chunk, score=score))
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:k]

    def save(self, path: str | Path) -> None:
        """Write the index to a JSON file (atomically, via a temp file)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "chunks": [c.model_dump() for c in self._chunks.values()],
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format version: {data.get('version')!r}")
        index = cls(k1=data["k1"], b=data["b"])
        index.add([Chunk.model_validate(c) for c in data["chunks"]])
        return index
