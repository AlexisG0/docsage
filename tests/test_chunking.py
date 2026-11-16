import pytest

from docsage.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_documents,
)
from docsage.loaders.base import Document

MD = """# Guide

Intro paragraph.

## Install

Run the installer. It takes a minute.

```
# not a heading
pip install docsage
```

## Usage

### CLI

Call the command.

## Install

Second install section.
"""


def doc(text: str, headings: list[str] | None = None) -> Document:
    return Document(text=text, path="a.md", title="T", headings=headings or [])


def test_fixed_offsets_match_text_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    chunks = FixedSizeChunker(chunk_size=60, overlap=15).chunk(doc(text))
    assert len(chunks) > 3
    for c in chunks:
        assert text[c.start : c.end] == c.text
        assert len(c.text) <= 60
    assert chunks[1].start < chunks[0].end  # overlap
    assert chunks[-1].end == len(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fixed_carries_metadata() -> None:
    d = Document(text="hello world", path="p.pdf", title="T", page=3)
    (c,) = FixedSizeChunker(100, 0).chunk(d)
    assert (c.path, c.title, c.page) == ("p.pdf", "T", 3)


def test_fixed_validates_args() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(0)
    with pytest.raises(ValueError):
        FixedSizeChunker(10, 10)


def test_fixed_empty_text() -> None:
    assert FixedSizeChunker().chunk(doc("")) == []


def test_recursive_tracks_heading_paths() -> None:
    d = doc(MD.strip())
    chunks = RecursiveChunker(max_chars=200).chunk(d)
    paths = [c.headings for c in chunks]
    assert ["Guide"] in paths
    assert ["Guide", "Install"] in paths
    assert ["Guide", "Usage", "CLI"] in paths
    assert paths.count(["Guide", "Install"]) == 2
    assert all("not a heading" not in h for p in paths for h in p)
    for c in chunks:
        assert d.text[c.start : c.end] == c.text


def test_recursive_respects_max_chars() -> None:
    long = "Sentence number one is here. " * 40 + "x" * 300
    chunks = RecursiveChunker(max_chars=100).chunk(doc("# H\n\n" + long))
    assert len(chunks) > 5
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.headings == ["H"] for c in chunks)


def test_recursive_merges_small_pieces() -> None:
    chunks = RecursiveChunker(max_chars=500).chunk(doc("# H\n\nOne.\n\nTwo.\n\nThree."))
    assert len(chunks) == 1
    assert "One." in chunks[0].text and "Three." in chunks[0].text


def test_recursive_plain_text_headings_from_html() -> None:
    d = doc("Setup\n\nDo this.\n\nUsage\n\nDo that.", headings=["Setup", "Usage"])
    chunks = RecursiveChunker(max_chars=50).chunk(d)
    assert [c.headings for c in chunks if "Do that" in c.text] == [["Usage"]]


def test_sentence_window_context() -> None:
    d = doc("One is first. Two is second. Three is third. Four is last.")
    chunks = SentenceWindowChunker(window=1).chunk(d)
    assert chunks[1].text == "Two is second."
    assert chunks[1].context == "One is first. Two is second. Three is third."
    assert chunks[0].context == "One is first. Two is second."
    for c in chunks:
        assert d.text[c.start : c.end] == c.text


def test_sentence_window_zero() -> None:
    chunks = SentenceWindowChunker(window=0).chunk(doc("A b. C d."))
    assert chunks[0].context == chunks[0].text


def test_chunk_documents_restarts_index() -> None:
    docs = [doc("A. B."), doc("C. D.")]
    chunks = chunk_documents(docs, SentenceWindowChunker())
    assert [c.index for c in chunks] == [0, 1, 0, 1]
