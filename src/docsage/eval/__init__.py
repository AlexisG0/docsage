"""Retrieval evaluation: a dataset format, ranking metrics and a runner."""

from __future__ import annotations

from docsage.eval.dataset import EvalExample, load_dataset
from docsage.eval.metrics import mrr, ndcg_at_k, recall_at_k
from docsage.eval.runner import EvalReport, evaluate

__all__ = [
    "EvalExample",
    "EvalReport",
    "evaluate",
    "load_dataset",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
