from pathlib import Path

import pytest

from docsage.chunking.base import Chunk
from docsage.embeddings import HashingEmbedder
from docsage.vectorstore import InMemoryVectorStore, VectorStore, chunk_id

TEXTS = [
    ("a.md", 0, "install the package with pip install docsage"),
    ("a.md", 1, "configure the api key through environment variables"),
    ("b.md", 0, "the cli exposes ingest and ask commands"),
    ("c.pdf", 0, "reciprocal rank fusion merges ranked lists"),
]


def make_chunks(texts: list[tuple[str, int, str]] = TEXTS) -> list[Chunk]:
    return [
        Chunk(
            text=t,
            path=p,
            index=i,
            start=0,
            end=len(t),
            title=p.upper(),
            page=2 if p.endswith("pdf") else None,
        )
        for p, i, t in texts
    ]


@pytest.fixture
def emb() -> HashingEmbedder:
    return HashingEmbedder()


def make_store(kind: str, path: Path, dimension: int) -> VectorStore:
    if kind == "memory":
        return InMemoryVectorStore(dimension)
    pytest.importorskip("lancedb")
    from docsage.vectorstore.lance import LanceVectorStore

    return LanceVectorStore(path, dimension)


@pytest.fixture(params=["memory", "lance"])
def store(request: pytest.FixtureRequest, tmp_path: Path, emb: HashingEmbedder) -> VectorStore:
    s = make_store(request.param, tmp_path / "db", emb.dimension)
    chunks = make_chunks()
    assert s.upsert(chunks, emb.embed([c.text for c in chunks])) == len(chunks)
    return s


def test_search_returns_best_match_first(store: VectorStore, emb: HashingEmbedder) -> None:
    hits = store.search(emb.embed(["reciprocal rank fusion"])[0], k=2)
    assert hits[0].chunk.path == "c.pdf"
    assert hits[0].chunk.page == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].id == chunk_id(hits[0].chunk)


def test_filters(store: VectorStore, emb: HashingEmbedder) -> None:
    q = emb.embed(["reciprocal rank fusion"])[0]
    hits = store.search(q, k=10, filters={"path": "a.md"})
    assert {h.chunk.path for h in hits} == {"a.md"}
    assert len(hits) == 2
    assert len(store.search(q, k=10, filters={"path": "a.md", "index": 1})) == 1
    assert store.search(q, filters={"path": "nope.md"}) == []
    with pytest.raises(ValueError):
        store.search(q, filters={"text": "x"})


def test_filter_value_is_escaped(store: VectorStore, emb: HashingEmbedder) -> None:
    q = emb.embed(["x"])[0]
    assert store.search(q, filters={"path": "a' OR '1'='1"}) == []


def test_upsert_replaces_existing(store: VectorStore, emb: HashingEmbedder) -> None:
    new = make_chunks([("b.md", 0, "completely different text about docker")])
    store.upsert(new, emb.embed([new[0].text]))
    assert store.count() == 4
    hit = store.search(emb.embed(["docker"])[0], k=1)[0]
    assert hit.chunk.text.endswith("docker")


def test_delete_by_path_and_ids(store: VectorStore) -> None:
    assert store.delete(path="a.md") == 2
    assert store.delete(ids=["b.md#p0#0"]) == 1
    assert store.delete(path="a.md") == 0
    assert store.delete() == 0
    assert store.count() == 1


def test_lance_persists_across_instances(tmp_path: Path, emb: HashingEmbedder) -> None:
    first = make_store("lance", tmp_path / "db", emb.dimension)
    chunks = make_chunks()
    first.upsert(chunks, emb.embed([c.text for c in chunks]))
    assert make_store("lance", tmp_path / "db", emb.dimension).count() == 4
    with pytest.raises(ValueError):
        make_store("lance", tmp_path / "db", emb.dimension + 1)


@pytest.mark.parametrize("kind", ["memory", "lance"])
def test_empty_store_and_bad_input(kind: str, tmp_path: Path, emb: HashingEmbedder) -> None:
    s = make_store(kind, tmp_path / "empty", emb.dimension)
    assert s.count() == 0
    assert s.search(emb.embed(["x"])[0]) == []
    assert s.upsert([], []) == 0
    with pytest.raises(ValueError):
        s.upsert(make_chunks()[:1], [[0.0] * 3])


def test_memory_ties_are_ordered_by_id(emb: HashingEmbedder) -> None:
    s = make_store("memory", Path("."), emb.dimension)
    chunks = make_chunks()[:2]
    vec = emb.embed(["same text"])[0]
    s.upsert(list(reversed(chunks)), [vec, vec])
    ids = [h.id for h in s.search(vec, k=2)]
    assert ids == sorted(ids)


def test_delete_without_criteria_is_a_noop(emb: HashingEmbedder) -> None:
    s = make_store("memory", Path("."), emb.dimension)
    chunks = make_chunks()
    s.upsert(chunks, emb.embed([c.text for c in chunks]))
    assert s.delete() == 0
    assert s.delete(ids=[]) == 0
    assert s.count() == len(chunks)
