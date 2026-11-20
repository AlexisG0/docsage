import sys
import types
from typing import Any

import pytest

from docsage.chunking.base import Chunk
from docsage.embeddings import HashingEmbedder
from docsage.keyword import BM25Index
from docsage.retrieval import CrossEncoderReranker, HybridRetriever, OverlapReranker, rerank
from docsage.vectorstore.base import SearchHit, chunk_id
from docsage.vectorstore.memory import InMemoryVectorStore


def chunk(path: str, index: int, text: str) -> Chunk:
    return Chunk(text=text, path=path, index=index, start=0, end=len(text))


def hit(path: str, text: str) -> SearchHit:
    c = chunk(path, 0, text)
    return SearchHit(id=chunk_id(c), chunk=c, score=0.0)


def test_rerank_orders_by_score_and_truncates() -> None:
    hits = [hit("a.md", "unrelated words"), hit("b.md", "set max_chars here")]
    out = rerank(OverlapReranker(), "set max_chars", hits, k=1)
    assert [h.chunk.path for h in out] == ["b.md"]
    assert out[0].score == 1.0


def test_rerank_ties_keep_incoming_order() -> None:
    hits = [hit("a.md", "nothing"), hit("b.md", "nothing")]
    out = rerank(OverlapReranker(), "query", hits)
    assert [h.chunk.path for h in out] == ["a.md", "b.md"]


def test_rerank_empty_and_bad_scorer() -> None:
    assert rerank(OverlapReranker(), "q", []) == []

    class Broken:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [1.0]

    with pytest.raises(ValueError):
        rerank(Broken(), "q", [hit("a.md", "x"), hit("b.md", "y")])


def test_retriever_applies_reranker_to_candidate_pool() -> None:
    chunks = [
        chunk("docs/a.md", 0, "alpha beta gamma"),
        chunk("docs/b.md", 0, "alpha only"),
        chunk("docs/c.md", 0, "gamma delta alpha beta"),
    ]
    embedder = HashingEmbedder(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    index = BM25Index()
    index.add(chunks)

    class Prefer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [1.0 if "delta" in p else 0.0 for p in passages]

    plain = HybridRetriever(store, index, embedder)
    reranked = HybridRetriever(store, index, embedder, reranker=Prefer())
    top = reranked.search("alpha beta", k=1)
    assert top[0].chunk.path == "docs/c.md"
    assert len(plain.search("alpha beta", k=3)) == len(reranked.search("alpha beta", k=3)) == 3


def test_cross_encoder_missing_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="docsage\\[local\\]"):
        CrossEncoderReranker()


def test_cross_encoder_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, name: str) -> None:
            self.name = name

        def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
            return [float(len(p)) for _, p in pairs]

    fake: Any = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    r = CrossEncoderReranker("any-model")
    assert r.score("q", ["a", "abc"]) == [1.0, 3.0]
    assert r.score("q", []) == []
