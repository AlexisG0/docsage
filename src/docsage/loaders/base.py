"""Shared document model for loaders."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A unit of loaded text plus the metadata needed to cite it later.

    Markdown and HTML files produce one document per file. PDFs produce one
    document per page so that citations can point at a page number.
    """

    text: str
    path: str
    title: str | None = None
    headings: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)


def read_text(path: Path) -> str:
    """Read a text file as UTF-8, tolerating a BOM."""
    return path.read_text(encoding="utf-8-sig")
