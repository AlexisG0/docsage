"""Run a retriever over a dataset and average the metrics."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from docsage.eval.dataset import EvalExample
from docsage.eval.metrics import mrr, ndcg_at_k, recall_at_k
from docsage.vectorstore import SearchHit


class Searcher(Protocol):
    def search(self, query: str, k: int = ...) -> list[SearchHit]: ...


class EvalReport(BaseModel):
    k: int
    questions: int
    recall: float
    mrr: float
    ndcg: float


def _is_relevant(path: str, relevant: list[str]) -> bool:
    return any(path == r or path.endswith("/" + r) for r in relevant)


def evaluate(retriever: Searcher, examples: list[EvalExample], k: int = 5) -> EvalReport:
    """Score ``retriever`` at cutoff ``k``.

    Relevance is judged per document: hits are collapsed to the first occurrence of each
    path so that several chunks of one document do not inflate recall or nDCG.
    """
    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    for ex in examples:
        hits = retriever.search(ex.question, k=k)
        seen: set[str] = set()
        paths: list[str] = []
        for h in hits:
            if h.chunk.path not in seen:
                seen.add(h.chunk.path)
                paths.append(h.chunk.path)
        flags = [_is_relevant(p, ex.relevant) for p in paths]
        total = len(ex.relevant)
        recalls.append(recall_at_k(flags, total, k))
        rrs.append(mrr(flags))
        ndcgs.append(ndcg_at_k(flags, total, k))
    n = len(examples)
    return EvalReport(
        k=k,
        questions=n,
        recall=sum(recalls) / n,
        mrr=sum(rrs) / n,
        ndcg=sum(ndcgs) / n,
    )
