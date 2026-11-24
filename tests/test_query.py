from typing import Any

from docsage.chunking.base import Chunk
from docsage.llm import ScriptedLLM
from docsage.retrieval import MultiQueryRetriever, expand_query, rewrite_query
from docsage.vectorstore.base import SearchHit


def hit(cid: str) -> SearchHit:
    return SearchHit(id=cid, chunk=Chunk(text=cid, path=cid, index=0, start=0, end=1), score=0.0)


class FakeSearcher:
    def __init__(self, table: dict[str, list[str]]) -> None:
        self.table = table
        self.seen: list[str] = []

    def search(
        self, query: str, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        self.seen.append(query)
        return [hit(i) for i in self.table.get(query, [])][:k]


def test_rewrite_takes_first_line_and_falls_back() -> None:
    assert rewrite_query(ScriptedLLM(['"how to set timeout"\nextra']), "q") == "how to set timeout"
    assert rewrite_query(ScriptedLLM(["  \n"]), "orig") == "orig"


def test_expand_dedupes_and_strips_bullets() -> None:
    llm = ScriptedLLM(["1. set timeout\n- Orig\n* configure timeout\n- third\n- fourth"])
    assert expand_query(llm, "orig", n=3) == ["orig", "set timeout", "configure timeout", "third"]
    assert expand_query(ScriptedLLM([]), "orig", n=0) == ["orig"]


def test_multi_query_fuses_rankings() -> None:
    base = FakeSearcher({"orig": ["a", "b"], "alt": ["c", "b"]})
    retriever = MultiQueryRetriever(base, ScriptedLLM(["alt"]), n_queries=1)
    assert [h.id for h in retriever.search("orig", k=3)] == ["b", "a", "c"]
    assert base.seen == ["orig", "alt"]
    assert retriever.search("orig", k=0) == []


def test_multi_query_with_rewrite() -> None:
    base = FakeSearcher({"clean": ["a"], "alt": ["b"]})
    retriever = MultiQueryRetriever(base, ScriptedLLM(["clean", "alt"]), n_queries=1, rewrite=True)
    assert [h.id for h in retriever.search("messy", k=2)] == ["a", "b"]
