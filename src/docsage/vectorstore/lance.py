"""LanceDB-backed vector store (embedded, on disk)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb

from docsage.chunking.base import Chunk
from docsage.vectorstore.base import FILTERABLE_FIELDS, SearchHit, chunk_id

TABLE = "chunks"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean filter values are not supported")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _quote(value)
    raise ValueError(f"unsupported filter value: {value!r}")


def build_where(filters: dict[str, Any]) -> str:
    """Translate an equality filter dict into a LanceDB SQL predicate."""
    parts = []
    for key, value in filters.items():
        if key not in FILTERABLE_FIELDS:
            raise ValueError(f"cannot filter on {key!r}; allowed: {', '.join(FILTERABLE_FIELDS)}")
        parts.append(f"`{key}` = {_literal(value)}")
    return " AND ".join(parts)


class LanceVectorStore:
    """Persists chunks and their vectors in a LanceDB table under ``directory``.

    Missing ``title`` and ``page`` are stored as ``""`` and ``0`` so they stay filterable.
    """

    def __init__(self, directory: str | Path, dimension: int) -> None:
        self.dimension = dimension
        self._db = lancedb.connect(str(directory))
        self._table: Any = None
        if TABLE in self._db.table_names():
            self._table = self._db.open_table(TABLE)
            stored = self._table.schema.field("vector").type.list_size
            if stored != dimension:
                raise ValueError(f"store holds {stored}-d vectors but dimension={dimension}")

    def _rows(self, chunks: list[Chunk], vectors: list[list[float]]) -> list[dict[str, Any]]:
        rows = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            if len(vec) != self.dimension:
                raise ValueError(f"vector has {len(vec)} dims, expected {self.dimension}")
            rows.append(
                {
                    "id": chunk_id(chunk),
                    "path": chunk.path,
                    "title": chunk.title or "",
                    "page": chunk.page or 0,
                    "index": chunk.index,
                    "payload": chunk.model_dump_json(),
                    "vector": vec,
                }
            )
        return rows

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0
        rows = self._rows(chunks, vectors)
        if self._table is None:
            self._table = self._db.create_table(TABLE, data=rows)
        else:
            (
                self._table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
        return len(rows)

    def delete(self, *, path: str | None = None, ids: list[str] | None = None) -> int:
        if self._table is None or (path is None and ids is None):
            return 0
        clauses = []
        if path is not None:
            clauses.append(f"path = {_quote(path)}")
        if ids is not None:
            if not ids:
                return 0
            clauses.append(f"id IN ({', '.join(_quote(i) for i in ids)})")
        before = self._table.count_rows()
        self._table.delete(" AND ".join(clauses))
        return int(before - self._table.count_rows())

    def search(
        self,
        vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if self._table is None or k <= 0:
            return []
        query = self._table.search(vector).metric("cosine").limit(k)
        if filters:
            query = query.where(build_where(filters), prefilter=True)
        return [
            SearchHit(
                id=row["id"],
                chunk=Chunk.model_validate_json(row["payload"]),
                score=1.0 - float(row["_distance"]),
            )
            for row in query.to_list()
        ]

    def count(self) -> int:
        return 0 if self._table is None else int(self._table.count_rows())
