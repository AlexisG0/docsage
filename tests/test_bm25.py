from pathlib import Path

import pytest

from docsage.chunking.base import Chunk
from docsage.keyword import BM25Index, tokenize


def chunk(path: str, index: int, text: str, page: int | None = None) -> Chunk:
    return Chunk(text=text, path=path, index=index, start=0, end=len(text), page=page)


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.add(
        [
            chunk("a.md", 0, "install the package with pip install docsage"),
            chunk("a.md", 1, "set max_chars to control the chunk size"),
            chunk("b.md", 0, "the cli exposes ingest and ask commands"),
            chunk("c.pdf", 0, "reciprocal rank fusion merges ranked lists", page=3),
        ]
    )
    return idx


def test_tokenize() -> None:
    assert tokenize("Set MAX_CHARS=800, then Run!") == ["set", "max_chars", "800", "then", "run"]


def test_exact_term_ranks_first(index: BM25Index) -> None:
    hits = index.search("max_chars")
    assert [h.chunk.path for h in hits] == ["a.md"]
    assert hits[0].chunk.index == 1
    assert hits[0].id == "a.md#p0#1"


def test_rarer_terms_and_repetition_score_higher(index: BM25Index) -> None:
    hits = index.search("install the")
    assert hits[0].id == "a.md#p0#0"
    assert hits[0].score >= hits[-1].score


def test_no_match_and_empty(index: BM25Index) -> None:
    assert index.search("kubernetes") == []
    assert index.search("!!!") == []
    assert index.search("install", k=0) == []
    assert BM25Index().search("install") == []


def test_filters(index: BM25Index) -> None:
    assert {h.chunk.path for h in index.search("the", filters={"path": "b.md"})} == {"b.md"}
    assert index.search("fusion", filters={"page": 3})[0].chunk.path == "c.pdf"
    assert index.search("fusion", filters={"page": 1}) == []
    with pytest.raises(ValueError):
        index.search("x", filters={"text": "x"})


def test_add_replaces_and_delete(index: BM25Index) -> None:
    index.add([chunk("b.md", 0, "completely different text about docker")])
    assert len(index) == 4
    assert index.search("ingest") == []
    assert index.search("docker")[0].chunk.path == "b.md"
    assert index.delete(path="a.md") == 2
    assert index.delete(ids=["b.md#p0#0"]) == 1
    assert index.delete(path="a.md") == 0
    assert index.delete() == 0
    assert len(index) == 1
    assert index.search("install") == []


def test_statistics_stay_consistent_after_delete(index: BM25Index) -> None:
    index.delete(path="a.md")
    fresh = BM25Index()
    fresh.add([chunk("b.md", 0, "the cli exposes ingest and ask commands")])
    fresh.add([chunk("c.pdf", 0, "reciprocal rank fusion merges ranked lists", page=3)])
    for q in ("ingest", "fusion lists"):
        assert [h.score for h in index.search(q)] == pytest.approx(
            [h.score for h in fresh.search(q)]
        )


def test_save_and_load_round_trip(index: BM25Index, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "bm25.json"
    index.save(target)
    loaded = BM25Index.load(target)
    assert len(loaded) == len(index)
    for q in ("install", "fusion", "the cli"):
        got, want = loaded.search(q), index.search(q)
        assert [h.id for h in got] == [h.id for h in want]
        assert [h.score for h in got] == pytest.approx([h.score for h in want])
    assert loaded.search("fusion")[0].chunk.page == 3


def test_load_rejects_unknown_version(tmp_path: Path) -> None:
    bad = tmp_path / "bm25.json"
    bad.write_text('{"version": 99}')
    with pytest.raises(ValueError):
        BM25Index.load(bad)


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Index(b=1.5)
