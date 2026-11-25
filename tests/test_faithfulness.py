import pytest

from docsage.chunking.base import Chunk
from docsage.faithfulness import (
    FaithfulnessReport,
    JudgeError,
    LLMJudge,
    check_faithfulness,
    parse_report,
)
from docsage.llm import ScriptedLLM
from docsage.vectorstore.base import SearchHit


def _hits() -> list[SearchHit]:
    chunk = Chunk(text="RRF merges ranked lists.", path="docs/rrf.md", index=0, start=0, end=24)
    return [SearchHit(id="docs/rrf.md#p0#0", chunk=chunk, score=0.9)]


def test_score_and_unsupported() -> None:
    reply = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
    report = parse_report(reply)
    assert report.score == 0.5
    assert report.unsupported == ["b"]


def test_empty_claims_score_one() -> None:
    assert FaithfulnessReport(claims=[]).score == 1.0


def test_parse_accepts_code_fence() -> None:
    report = parse_report('```json\n{"claims": [{"claim": "a", "supported": true}]}\n```')
    assert report.score == 1.0


@pytest.mark.parametrize("reply", ["not json", '{"claims": "x"}', "[]"])
def test_parse_rejects_garbage(reply: str) -> None:
    with pytest.raises(JudgeError):
        parse_report(reply)


def test_llm_judge_sends_passages_and_answer() -> None:
    llm = ScriptedLLM(['{"claims": [{"claim": "RRF merges lists", "supported": true}]}'])
    report = check_faithfulness(LLMJudge(llm), "RRF merges lists [1].", _hits())
    assert report.score == 1.0
    user = llm.calls[0][1].content
    assert "[1] RRF merges ranked lists." in user
    assert "RRF merges lists [1]." in user


def test_no_hits_skips_judge() -> None:
    llm = ScriptedLLM([])
    report = check_faithfulness(LLMJudge(llm), "Something.", [])
    assert report.score == 0.0
    assert llm.calls == []
