"""Query rewriting and multi-query expansion on top of any retriever."""

from __future__ import annotations

import re
from typing import Any, Protocol

from docsage.llm.base import LLM, Message
from docsage.retrieval.hybrid import reciprocal_rank_fusion
from docsage.vectorstore.base import SearchHit

_REWRITE_SYSTEM = (
    "Rewrite the user's question as a standalone search query for a technical documentation "
    "index. Keep identifiers, flags and error messages verbatim. Reply with the query only."
)
_EXPAND_SYSTEM = (
    "Write {n} alternative search queries for the user's question, using different wording "
    "and synonyms. Reply with one query per line and nothing else."
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class Searcher(Protocol):
    def search(
        self, query: str, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...


def rewrite_query(llm: LLM, query: str) -> str:
    """Return a standalone search query; falls back to the original if the reply is empty."""
    reply = llm.complete(
        [Message("system", _REWRITE_SYSTEM), Message("user", query)], max_tokens=128
    )
    lines = [ln.strip().strip('"') for ln in reply.splitlines() if ln.strip()]
    return lines[0] if lines else query


def expand_query(llm: LLM, query: str, n: int = 3) -> list[str]:
    """Return ``query`` followed by up to ``n`` distinct LLM-written variants."""
    if n <= 0:
        return [query]
    reply = llm.complete(
        [Message("system", _EXPAND_SYSTEM.format(n=n)), Message("user", query)], max_tokens=256
    )
    queries = [query]
    seen = {query.casefold()}
    for line in reply.splitlines():
        text = _BULLET.sub("", line).strip().strip('"')
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            queries.append(text)
        if len(queries) > n:
            break
    return queries


class MultiQueryRetriever:
    """Searches with the original query plus LLM variants and fuses the rankings with RRF."""

    def __init__(
        self,
        retriever: Searcher,
        llm: LLM,
        *,
        n_queries: int = 3,
        rewrite: bool = False,
        rrf_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.n_queries = n_queries
        self.rewrite = rewrite
        self.rrf_k = rrf_k

    def queries(self, query: str) -> list[str]:
        base = rewrite_query(self.llm, query) if self.rewrite else query
        return expand_query(self.llm, base, self.n_queries)

    def search(
        self, query: str, k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        rankings = [self.retriever.search(q, k=k, filters=filters) for q in self.queries(query)]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:k]
