"""Heading-aware recursive splitting."""

from __future__ import annotations

import re

from docsage.chunking.base import Chunk, Piece, make_chunks
from docsage.chunking.sentences import split_sentences
from docsage.loaders.base import Document

_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")

Span = tuple[str, int, int]  # text, start, end


def _sections(doc: Document) -> list[tuple[list[str], int, int]]:
    """Split a document into (heading path, start, end) sections.

    Markdown ATX headings give nested paths. Plain-text lines that exactly
    match a known document heading (as produced by the HTML loader) are
    treated as top-level headings.
    """
    known = set(doc.headings)
    text = doc.text
    marks: list[tuple[int, int, str]] = []  # offset, level, title
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        f = _FENCE.match(bare)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
        elif fence is None:
            h = _ATX.match(bare)
            if h and h.group(2):
                marks.append((offset, len(h.group(1)), h.group(2).strip()))
            elif bare.strip() in known:
                marks.append((offset, 1, bare.strip()))
        offset += len(line)

    sections: list[tuple[list[str], int, int]] = []
    stack: list[tuple[int, str]] = []
    prev = 0
    for off, level, title in marks:
        if off > prev:
            sections.append(([t for _, t in stack], prev, off))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        prev = off
    sections.append(([t for _, t in stack], prev, len(text)))
    return sections


def _split_paragraphs(text: str, base: int) -> list[Span]:
    return [
        (m.group(0), base + m.start(), base + m.end())
        for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.DOTALL)
    ]


def _split_words(text: str, base: int, size: int) -> list[Span]:
    pieces: list[Span] = []
    cur_start: int | None = None
    cur_end = 0
    for m in re.finditer(r"\S+", text):
        w_start, w_end = m.start(), m.end()
        # Words longer than the limit are hard-cut.
        while w_end - w_start > size:
            if cur_start is not None:
                pieces.append((text[cur_start:cur_end], base + cur_start, base + cur_end))
                cur_start = None
            pieces.append((text[w_start : w_start + size], base + w_start, base + w_start + size))
            w_start += size
        if cur_start is None:
            cur_start, cur_end = w_start, w_end
        elif w_end - cur_start <= size:
            cur_end = w_end
        else:
            pieces.append((text[cur_start:cur_end], base + cur_start, base + cur_end))
            cur_start, cur_end = w_start, w_end
    if cur_start is not None:
        pieces.append((text[cur_start:cur_end], base + cur_start, base + cur_end))
    return pieces


def _atoms(text: str, base: int, size: int) -> list[Span]:
    """Break text into units no longer than ``size``: paragraphs, sentences, then words."""
    out: list[Span] = []
    for para, p_start, _ in _split_paragraphs(text, base):
        if len(para) <= size:
            out.append((para, p_start, p_start + len(para)))
            continue
        for sent, s_start, s_end in split_sentences(para):
            if len(sent) <= size:
                out.append((sent, p_start + s_start, p_start + s_end))
            else:
                out.extend(_split_words(sent, p_start + s_start, size))
    return out


class RecursiveChunker:
    """Split on headings first, then paragraphs, sentences and words.

    Each chunk stays inside one section and carries that section's heading
    path. Small neighbouring pieces of a section are merged up to
    ``max_chars``, so a chunk never mixes headings.
    """

    def __init__(self, max_chars: int = 800) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        pieces: list[Piece] = []
        for path, start, end in _sections(doc):
            cur: tuple[int, int] | None = None
            for _, a_start, a_end in _atoms(text[start:end], start, self.max_chars):
                if cur is None:
                    cur = (a_start, a_end)
                elif a_end - cur[0] <= self.max_chars:
                    cur = (cur[0], a_end)
                else:
                    pieces.append((text[cur[0] : cur[1]], cur[0], cur[1], path, None))
                    cur = (a_start, a_end)
            if cur is not None:
                pieces.append((text[cur[0] : cur[1]], cur[0], cur[1], path, None))
        return make_chunks(doc, pieces)
