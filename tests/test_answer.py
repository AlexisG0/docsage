from docsage.answer import answer_question, build_messages, parse_citations
from docsage.chunking.base import Chunk
from docsage.llm import ScriptedLLM
from docsage.vectorstore.base import SearchHit


def hit(i: int, text: str = "some text", context: str | None = None) -> SearchHit:
    chunk = Chunk(
        text=text, path=f"d{i}.md", index=0, start=0, end=len(text), title=f"T{i}", context=context
    )
    return SearchHit(id=f"d{i}", chunk=chunk, score=1.0)


def test_parse_citations_dedupes_and_drops_invalid() -> None:
    assert parse_citations("a [2] b [1][2] c [9] [0]", 3) == [2, 1]


def test_build_messages_numbers_passages_and_prefers_context() -> None:
    msgs = build_messages("q?", [hit(1, "short", "wide window"), hit(2)])
    assert msgs[0].role == "system"
    assert "[1] (d1.md)\nwide window" in msgs[1].content
    assert "[2] (d2.md)\nsome text" in msgs[1].content
    assert msgs[1].content.endswith("Question: q?")


def test_answer_returns_only_cited_sources() -> None:
    llm = ScriptedLLM(["Yes [2]. Also [2][5]."])
    ans = answer_question(llm, "q?", [hit(1), hit(2, "x " * 300)])
    assert [s.number for s in ans.sources] == [2]
    assert ans.sources[0].path == "d2.md"
    assert ans.sources[0].title == "T2"
    assert len(ans.sources[0].snippet) <= 240
    assert len(llm.calls) == 1


def test_answer_without_hits_skips_llm() -> None:
    llm = ScriptedLLM([])
    ans = answer_question(llm, "q?", [])
    assert ans.sources == []
    assert llm.calls == []
