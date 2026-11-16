"""HTML loader built on the standard library parser."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from docsage.loaders.base import Document, read_text

_SKIP = {"script", "style", "noscript", "template", "head", "nav", "footer"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = _HEADINGS | {
    "p", "div", "section", "article", "main", "li", "ul", "ol", "pre", "tr",
    "table", "blockquote", "br", "dt", "dd",
}  # fmt: skip
_LINES = {"li", "br", "tr", "dt", "dd"}  # single line break, no blank line


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.headings: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._heading: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag in _BLOCKS:
            self.parts.append("\n" if tag in _LINES else "\n\n")
            if tag in _HEADINGS:
                self._heading = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCKS:
            if tag not in _LINES:
                self.parts.append("\n\n")
            if tag in _HEADINGS and self._heading is not None:
                text = " ".join("".join(self._heading).split())
                if text:
                    self.headings.append(text)
                self._heading = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._heading is not None:
            self._heading.append(data)


def _normalize(parts: list[str]) -> str:
    text = "".join(parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_html(path: str | Path) -> Document:
    """Load an HTML file as plain text.

    Scripts, styles and navigation chrome are dropped. The title comes from the
    ``<title>`` element, then the first heading, then the file name.
    """
    p = Path(path)
    parser = _Extractor()
    parser.feed(read_text(p))
    parser.close()
    title = " ".join("".join(parser.title_parts).split()) or None
    if title is None:
        title = parser.headings[0] if parser.headings else p.stem
    return Document(
        text=_normalize(parser.parts), path=str(p), title=title, headings=parser.headings
    )
