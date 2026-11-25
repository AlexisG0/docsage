"""Answer faithfulness: does every claim in an answer follow from the retrieved passages?"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from docsage.llm.base import LLM, Message
from docsage.vectorstore.base import SearchHit

JUDGE_PROMPT = (
    "You check an answer against numbered context passages. Split the answer into its "
    "individual factual claims and decide for each whether the passages support it. "
    'Reply with JSON only, shaped like {"claims": [{"claim": "...", "supported": true}]}. '
    "Do not use outside knowledge."
)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class FaithfulnessReport(BaseModel):
    claims: list[ClaimVerdict]

    @property
    def score(self) -> float:
        """Share of supported claims; 1.0 when the answer makes no claims."""
        if not self.claims:
            return 1.0
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


@runtime_checkable
class Judge(Protocol):
    """Anything that can grade an answer against its context."""

    def judge(self, answer: str, passages: list[str]) -> FaithfulnessReport: ...


class JudgeError(RuntimeError):
    """The judge model returned something that is not a valid verdict."""


class LLMJudge:
    """Judge backed by any ``LLM``; the model must reply with the JSON shape in the prompt."""

    def __init__(self, llm: LLM, *, max_tokens: int = 1024) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def judge(self, answer: str, passages: list[str]) -> FaithfulnessReport:
        context = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, 1))
        messages = [
            Message("system", JUDGE_PROMPT),
            Message("user", f"Context:\n\n{context}\n\nAnswer:\n{answer}"),
        ]
        reply = self._llm.complete(messages, temperature=0.0, max_tokens=self._max_tokens)
        return parse_report(reply)


def parse_report(reply: str) -> FaithfulnessReport:
    """Parse a judge reply, tolerating a markdown code fence around the JSON."""
    text = reply.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return FaithfulnessReport.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JudgeError(f"unparseable judge reply: {reply[:200]!r}") from exc


def check_faithfulness(judge: Judge, answer: str, hits: list[SearchHit]) -> FaithfulnessReport:
    """Grade ``answer`` against the text of ``hits``.

    Without passages nothing can support the answer, so the judge is not called and the
    answer counts as one unsupported claim.
    """
    if not hits:
        return FaithfulnessReport(claims=[ClaimVerdict(claim=answer, supported=False)])
    return judge.judge(answer, [h.chunk.context or h.chunk.text for h in hits])
