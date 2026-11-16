"""Markdown loader."""

from __future__ import annotations

import re
from pathlib import Path

from docsage.loaders.base import Document, read_text

_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def _front_matter_title(block: str) -> str | None:
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "title":
            return value.strip().strip("'\"") or None
    return None


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Return (level, title) for each ATX heading, ignoring fenced code blocks."""
    headings: list[tuple[int, str]] = []
    fence: str | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        h = _ATX.match(line)
        if h and h.group(2):
            headings.append((len(h.group(1)), h.group(2).strip()))
    return headings


def load_markdown(path: str | Path) -> Document:
    """Load a Markdown file.

    The title comes from the front matter ``title`` key, then the first level-1
    heading, then the file name. Front matter is stripped from the text.
    """
    p = Path(path)
    text = read_text(p)
    title: str | None = None
    m = _FRONT_MATTER.match(text)
    if m:
        title = _front_matter_title(m.group(1))
        text = text[m.end() :]
    headings = extract_headings(text)
    if title is None:
        title = next((t for level, t in headings if level == 1), p.stem)
    return Document(text=text.strip(), path=str(p), title=title, headings=[t for _, t in headings])
