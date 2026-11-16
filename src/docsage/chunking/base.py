"""Chunk model and the chunker interface."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from docsage.loaders.base import Document


class Chunk(BaseModel):
    """A piece of a document, with the metadata needed to cite it.

    ``start`` and ``end`` are character offsets into the source document text.
    ``context`` is optional wider text (used by the sentence-window strategy):
    ``text`` is what gets embedded, ``context`` is what gets shown to the LLM.
    """

    text: str
    path: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    title: str | None = None
    headings: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    context: str | None = None


class Chunker(Protocol):
    """Anything that can turn a document into chunks."""

    def chunk(self, doc: Document) -> list[Chunk]: ...


Piece = tuple[str, int, int, list[str], str | None]  # text, start, end, headings, context


def make_chunks(doc: Document, pieces: list[Piece]) -> list[Chunk]:
    """Build numbered chunks from (text, start, end, headings, context) tuples."""
    return [
        Chunk(
            text=text,
            path=doc.path,
            index=i,
            start=start,
            end=end,
            title=doc.title,
            headings=headings,
            page=doc.page,
            context=context,
        )
        for i, (text, start, end, headings, context) in enumerate(pieces)
    ]
