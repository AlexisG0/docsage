"""Sentence-window chunking."""

from __future__ import annotations

import re

from docsage.chunking.base import Chunk, Piece, make_chunks
from docsage.loaders.base import Document

# A sentence ends at ./!/? followed by whitespace, or at a blank line.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start, end) spans."""
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for m in [*_BOUNDARY.finditer(text), None]:
        end = m.start() if m else len(text)
        raw = text[pos:end]
        s = raw.strip()
        if s:
            start = pos + len(raw) - len(raw.lstrip())
            spans.append((s, start, start + len(s)))
        if m:
            pos = m.end()
    return spans


class SentenceWindowChunker:
    """One chunk per sentence; ``context`` holds the sentence plus its neighbours.

    The short sentence is embedded for precise matching, and the surrounding
    window is returned to the LLM for context.
    """

    def __init__(self, window: int = 2) -> None:
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = window

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        pieces: list[Piece] = []
        for i, (s, start, end) in enumerate(sents):
            lo = max(0, i - self.window)
            hi = min(len(sents), i + self.window + 1)
            context = doc.text[sents[lo][1] : sents[hi - 1][2]]
            pieces.append((s, start, end, [], context))
        return make_chunks(doc, pieces)
