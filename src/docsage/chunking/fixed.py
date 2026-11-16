"""Fixed-size chunking with overlap."""

from __future__ import annotations

from docsage.chunking.base import Chunk, Piece, make_chunks
from docsage.loaders.base import Document


class FixedSizeChunker:
    """Slide a window of ``chunk_size`` characters, stepping by size minus overlap.

    Window edges are moved back to the nearest whitespace when possible so
    words are not cut in half.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        pieces: list[Piece] = []
        pos = 0
        n = len(text)
        while pos < n:
            end = min(pos + self.chunk_size, n)
            if end < n:
                cut = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
                # Only back off if that keeps more than half the window.
                if cut > pos + self.chunk_size // 2:
                    end = cut
            piece = text[pos:end]
            stripped = piece.strip()
            if stripped:
                lead = len(piece) - len(piece.lstrip())
                pieces.append((stripped, pos + lead, pos + lead + len(stripped), [], None))
            if end >= n:
                break
            pos = max(end - self.overlap, pos + 1)
        return make_chunks(doc, pieces)
