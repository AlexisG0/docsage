"""Grounded answer generation: numbered context, inline ``[n]`` citations, source snippets."""

from __future__ import annotations

import re

from pydantic import BaseModel

from docsage.llm.base import LLM, Message
from docsage.vectorstore.base import SearchHit

SYSTEM_PROMPT = (
    "You answer questions about technical documentation using only the numbered context "
    "passages provided. Cite every claim with the passage number in square brackets, "
    "like [1] or [2][3]. If the context does not contain the answer, say that you do not "
    "know and do not cite anything."
)

_CITATION = re.compile(r"\[(\d+)\]")
_SNIPPET_CHARS = 240


class Source(BaseModel):
    """A passage the answer cites, numbered as in the answer text."""

    number: int
    path: str
    title: str | None = None
    page: int | None = None
    snippet: str


class Answer(BaseModel):
    text: str
    sources: list[Source]


def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
    """Build the chat prompt: passages are numbered from 1 in retrieval order."""
    context = "\n\n".join(
        f"[{n}] ({h.chunk.path})\n{h.chunk.context or h.chunk.text}" for n, h in enumerate(hits, 1)
    )
    user = f"Context:\n\n{context}\n\nQuestion: {question}" if hits else f"Question: {question}"
    return [Message("system", SYSTEM_PROMPT), Message("user", user)]


def parse_citations(text: str, max_number: int) -> list[int]:
    """Cited passage numbers in order of first appearance; numbers outside 1..max are dropped."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= max_number and n not in seen:
            seen.append(n)
    return seen


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[: _SNIPPET_CHARS - 1].rstrip() + "…"


def answer_question(
    llm: LLM,
    question: str,
    hits: list[SearchHit],
    *,
    max_tokens: int = 1024,
) -> Answer:
    """Ask ``llm`` to answer from ``hits`` and return the text with the passages it cited.

    With no hits the model is not called: there is nothing to ground an answer on.
    """
    if not hits:
        return Answer(text="I could not find anything relevant in the documentation.", sources=[])
    reply = llm.complete(build_messages(question, hits), temperature=0.0, max_tokens=max_tokens)
    sources = []
    for n in parse_citations(reply, len(hits)):
        chunk = hits[n - 1].chunk
        sources.append(
            Source(
                number=n,
                path=chunk.path,
                title=chunk.title,
                page=chunk.page,
                snippet=_snippet(chunk.text),
            )
        )
    return Answer(text=reply.strip(), sources=sources)
