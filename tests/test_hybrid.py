import pytest

from docsage.chunking.base import Chunk
from docsage.embeddings import HashingEmbedder
from docsage.keyword import BM25Index
from docsage.retrieval import HybridRetriever, reciprocal_rank_fusion
from docsage.vectorstore.base import SearchHit
from docsage.vectorstore.memory import InMemoryVectorStore


def chunk(path: str, index: int, text: str) -> Chunk:
    return Chunk(text=text, path=path, index=index, start=0, end=len(text))


def hit(cid: str) -> SearchHit:
    path, _, idx = cid.partition(":")
    return SearchHit(id=cid, chunk=chunk(path, int(idx), cid), score=0.0)


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.id for h in hits]


def test_rrf_rewards_agreement() -> None:
    a = [hit("x:0"), hit("y:0"), hit("z:0")]
    b = [hit("y:0"), hit("x:0"), hit("w:0")]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert ids(fused)[:2] == ["x:0", "y:0"]  # tie on score, broken by id
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert set(ids(fused)) == {"x:0", "y:0", "z:0", "w:0"}


def test_rrf_weights_shift_the_order() -> None:
    a = [hit("a:0")]
    b = [hit("b:0")]
    assert ids(reciprocal_rank_fusion([a, b], weights=[1, 3]))[0] == "b:0"
    assert ids(reciprocal_rank_fusion([a, b], weights=[3, 1]))[0] == "a:0"


def test_rrf_validation() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a:0")]], weights=[1, 2])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a:0")]], weights=[-1])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a:0")]], k=-1)
    assert reciprocal_rank_fusion([]) == []


@pytest.fixture
def retriever() -> HybridRetriever:
    chunks = [
        chunk("docs/chunking.md", 0, "Set max_chars to control the size of each chunk."),
        chunk("docs/chunking.md", 1, "Overlap keeps context between neighbouring chunks."),
        chunk("docs/search.md", 0, "Hybrid search combines keyword and vector retrieval."),
        chunk("docs/install.md", 0, "Install the package with uv and run the tests."),
    ]
    embedder = HashingEmbedder(dimension=128)
    store = InMemoryVectorStore(dimension=128)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    index = BM25Index()
    index.add(chunks)
    return HybridRetriever(store, index, embedder)


def test_exact_identifier_is_found(retriever: HybridRetriever) -> None:
    hits = retriever.search("max_chars", k=2)
    assert hits[0].id == "docs/chunking.md#p0#0"
    assert len(hits) <= 2


def test_semantic_side_contributes(retriever: HybridRetriever) -> None:
    # No shared term with the keyword side beyond "search", but vectors still rank.
    hits = retriever.search("hybrid keyword vector retrieval", k=1)
    assert hits[0].chunk.path == "docs/search.md"


def test_filters_apply_to_both_sides(retriever: HybridRetriever) -> None:
    hits = retriever.search("chunk", k=5, filters={"path": "docs/search.md"})
    assert hits
    assert {h.chunk.path for h in hits} == {"docs/search.md"}


def test_zero_weight_disables_a_retriever(retriever: HybridRetriever) -> None:
    retriever.vector_weight = 0
    # Keyword-only: a query with no lexical match returns nothing.
    assert retriever.search("zzzz", k=3) == []
    retriever.vector_weight = 1
    retriever.keyword_weight = 0
    assert retriever.search("zzzz", k=3)  # vectors always return neighbours


def test_k_and_validation(retriever: HybridRetriever) -> None:
    assert retriever.search("chunk", k=0) == []
    assert len(retriever.search("chunk", k=2)) == 2
    with pytest.raises(ValueError):
        HybridRetriever(
            retriever.store, retriever.index, retriever.embedder, keyword_weight=0, vector_weight=0
        )
