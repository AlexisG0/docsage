"""Document loaders that keep source metadata."""

from __future__ import annotations

from pathlib import Path

from docsage.loaders.base import Document
from docsage.loaders.html import load_html
from docsage.loaders.markdown import load_markdown
from docsage.loaders.pdf import load_pdf

_MARKDOWN = {".md", ".markdown"}
_HTML = {".html", ".htm"}


def load_file(path: str | Path) -> list[Document]:
    """Load any supported file, dispatching on its extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _MARKDOWN:
        return [load_markdown(p)]
    if suffix in _HTML:
        return [load_html(p)]
    if suffix == ".pdf":
        return load_pdf(p)
    raise ValueError(f"unsupported file type: {p.suffix or p.name}")


__all__ = ["Document", "load_file", "load_html", "load_markdown", "load_pdf"]
