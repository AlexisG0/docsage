"""PDF loader: one document per page."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from docsage.loaders.base import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF as one Document per non-empty page (1-based ``page``).

    The title comes from the PDF metadata when present, otherwise the file name.
    """
    p = Path(path)
    reader = PdfReader(p)
    meta_title = reader.metadata.title if reader.metadata else None
    title = (meta_title or "").strip() or p.stem
    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, path=str(p), title=title, page=number))
    return docs
