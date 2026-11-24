"""A small on-disk index: BM25 plus vectors, kept under one directory.

The vector side is held in memory and saved as JSON next to the BM25 file, which is plenty for
documentation-sized corpora and needs no optional dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from docsage.chunking.base import Chunk
from docsage.embeddings import Embedder
from docsage.keyword import BM25Index
from docsage.retrieval import HybridRetriever
from docsage.vectorstore import InMemoryVectorStore

_BM25_FILE = "bm25.json"
_VECTORS_FILE = "vectors.json"


class Workspace:
    def __init__(self, home: str | Path, embedder: Embedder) -> None:
        self.home = Path(home)
        self.embedder = embedder
        self.index = BM25Index()
        self.store = InMemoryVectorStore(embedder.dimension)

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
        return ws

    def replace(self, path: str, chunks: list[Chunk]) -> None:
        """Drop everything stored for ``path`` and index ``chunks`` in its place."""
        self.index.delete(path=path)
        self.store.delete(path=path)
        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            self.index.add(chunks)
            self.store.upsert(chunks, vectors)

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.index.save(self.home / _BM25_FILE)
        rows = [
            {"chunk": chunk.model_dump(), "vector": vec} for chunk, vec in self.store._rows.values()
        ]
        payload = {"dimension": self.embedder.dimension, "rows": rows}
        tmp = self.home / (_VECTORS_FILE + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.home / _VECTORS_FILE)

    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.store, self.index, self.embedder)

    def stats(self) -> dict[str, int]:
        paths = {chunk.path for chunk, _ in self.store._rows.values()}
        return {"documents": len(paths), "chunks": self.store.count()}
