import math
import sys

import pytest

from docsage.embeddings import (
    Embedder,
    HashingEmbedder,
    cosine,
    embed_query,
    load_embedder,
)


def test_deterministic_and_normalised() -> None:
    e = HashingEmbedder(64)
    a = e.embed(["Install the package with pip"])[0]
    b = HashingEmbedder(64).embed(["Install the package with pip"])[0]
    assert a == b
    assert len(a) == e.dimension == 64
    assert math.isclose(sum(x * x for x in a), 1.0)


def test_similarity_reflects_shared_words() -> None:
    e = HashingEmbedder()
    q = embed_query(e, "how to configure logging")
    near = embed_query(e, "configure logging in the app")
    far = embed_query(e, "database migration rollback")
    assert cosine(q, near) > cosine(q, far)


def test_empty_text_and_batch() -> None:
    e = HashingEmbedder(8)
    assert e.embed([]) == []
    assert e.embed([""]) == [[0.0] * 8]


def test_invalid_dimension_and_cosine_mismatch() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(0)
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_protocol_and_factory() -> None:
    assert isinstance(load_embedder("hashing", dimension=16), Embedder)
    with pytest.raises(ValueError, match="unknown embedder"):
        load_embedder("nope")


def test_sentence_transformers_missing_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"docsage\[local\]"):
        load_embedder("sentence-transformers")


def test_hashing_ignores_case_and_punctuation() -> None:
    e = HashingEmbedder(64)
    assert e.embed(["Hello, World!"]) == e.embed(["hello world"])


def test_sentence_transformers_backend_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            calls.append(kwargs)
            return [[1.0, 0.0, 0.0] for _ in texts]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    e = load_embedder("sentence-transformers", model_name="tiny", batch_size=4)
    assert e.dimension == 3
    assert e.embed([]) == []
    assert e.embed(["a", "b"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert calls[0]["batch_size"] == 4
    assert calls[0]["normalize_embeddings"] is True
