from pathlib import Path

from docsage.chunking.base import Chunk
from docsage.embeddings import HashingEmbedder
from docsage.workspace import Workspace, content_hash


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return super().embed(texts)


def _chunk(text: str, path: str = "a.md", index: int = 0) -> Chunk:
    return Chunk(text=text, path=path, index=index, start=0, end=len(text))


def test_unchanged_chunks_are_not_embedded_again(tmp_path: Path) -> None:
    emb = CountingEmbedder()
    ws = Workspace(tmp_path, emb)
    ws.replace("a.md", [_chunk("alpha"), _chunk("beta", index=1)])
    ws.replace("a.md", [_chunk("alpha"), _chunk("gamma", index=1)])
    assert emb.seen == ["alpha", "beta", "gamma"]


def test_cache_survives_save_and_reopen(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, CountingEmbedder())
    ws.replace("a.md", [_chunk("alpha")], digest="d1")
    ws.save()

    emb = CountingEmbedder()
    again = Workspace.open(tmp_path, emb)
    assert again.is_current("a.md", "d1")
    assert not again.is_current("a.md", "d2")
    again.replace("b.md", [_chunk("alpha", path="b.md")])
    assert emb.seen == []


def test_remove_drops_document_and_hash(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, HashingEmbedder())
    ws.replace("a.md", [_chunk("alpha")], digest="d1")
    ws.remove("a.md")
    assert ws.paths() == []
    assert not ws.is_current("a.md", "d1")
    assert ws.retriever() is not None


def test_content_hash_separates_parts() -> None:
    assert content_hash("ab", "c") != content_hash("a", "bc")
