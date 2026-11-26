"""A small on-disk index: BM25 plus vectors, kept under one directory.

The vector side is held in memory and saved as JSON next to the BM25 file, which is plenty for
documentation-sized corpora and needs no optional dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docsage.chunking.base import Chunk
from docsage.embeddings import Embedder
from docsage.keyword import BM25Index
from docsage.retrieval import HybridRetriever
from docsage.vectorstore import InMemoryVectorStore

_BM25_FILE = "bm25.json"
_VECTORS_FILE = "vectors.json"
_HASHES_FILE = "hashes.json"


def content_hash(*parts: str | bytes) -> str:
    """Stable digest of the inputs that decide what a document's chunks look like."""
    h = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else part
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


class Workspace:
    def __init__(self, home: str | Path, embedder: Embedder) -> None:
        self.home = Path(home)
        self.embedder = embedder
        self.index = BM25Index()
        self.store = InMemoryVectorStore(embedder.dimension)
        self.hashes: dict[str, str] = {}
        # Vectors by text digest, so unchanged chunks are never embedded twice.
        self._cache: dict[str, list[float]] = {}

    @classmethod
    def open(cls, home: str | Path, embedder: Embedder) -> Workspace:
        ws = cls(home, embedder)
        bm25 = ws.home / _BM25_FILE
        vectors = ws.home / _VECTORS_FILE
        if bm25.exists() and vectors.exists():
            ws.index = BM25Index.load(bm25)
            data = json.loads(vectors.read_text(encoding="utf-8"))
            if data["dimension"] != embedder.dimension:
                raise ValueError(
                    f"index was built with {data['dimension']}-dim vectors but the embedder "
                    f"produces {embedder.dimension}; re-ingest or pick the same embedder"
                )
            rows = data["rows"]
            ws.store.upsert(
                [Chunk.model_validate(r["chunk"]) for r in rows], [r["vector"] for r in rows]
            )
            ws._cache = {content_hash(r["chunk"]["text"]): r["vector"] for r in rows}
            hashes = ws.home / _HASHES_FILE
            if hashes.exists():
                ws.hashes = json.loads(hashes.read_text(encoding="utf-8"))
        return ws

    def is_current(self, path: str, digest: str) -> bool:
        """True if ``path`` is indexed from content with this digest."""
        return self.hashes.get(path) == digest

    def paths(self) -> list[str]:
        return sorted({chunk.path for chunk, _ in self.store._rows.values()})

    def remove(self, path: str) -> None:
        """Delete a document from the index."""
        self.index.delete(path=path)
        self.store.delete(path=path)
        self.hashes.pop(path, None)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [content_hash(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if missing:
            vectors = self.embedder.embed(list(missing.values()))
            self._cache.update(zip(missing, vectors, strict=True))
        return [self._cache[k] for k in keys]

    def replace(self, path: str, chunks: list[Chunk], digest: str | None = None) -> None:
        """Drop everything stored for ``path`` and index ``chunks`` in its place.

        Passing ``digest`` records it so a later run can skip the document with ``is_current``.
        """
        self.remove(path)
        if chunks:
            vectors = self._embed([c.text for c in chunks])
            self.index.add(chunks)
            self.store.upsert(chunks, vectors)
        if digest is not None:
            self.hashes[path] = digest

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.index.save(self.home / _BM25_FILE)
        rows = [
            {"chunk": chunk.model_dump(), "vector": vec} for chunk, vec in self.store._rows.values()
        ]
        payload = {"dimension": self.embedder.dimension, "rows": rows}
        (self.home / _HASHES_FILE).write_text(json.dumps(self.hashes), encoding="utf-8")
        tmp = self.home / (_VECTORS_FILE + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.home / _VECTORS_FILE)

    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.store, self.index, self.embedder)

    def stats(self) -> dict[str, int]:
        paths = {chunk.path for chunk, _ in self.store._rows.values()}
        return {"documents": len(paths), "chunks": self.store.count()}
