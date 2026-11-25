"""Ranking metrics over a list of binary relevance flags (best-ranked first)."""

from __future__ import annotations

import math


def recall_at_k(flags: list[bool], total_relevant: int, k: int) -> float:
    """Share of the relevant items found in the top ``k``."""
    if total_relevant <= 0 or k <= 0:
        return 0.0
    return sum(flags[:k]) / total_relevant


def mrr(flags: list[bool]) -> float:
    """Reciprocal rank of the first relevant item, 0 if there is none."""
    for rank, hit in enumerate(flags, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(flags: list[bool], total_relevant: int, k: int) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    if total_relevant <= 0 or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(flags[:k]) if hit)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    return dcg / ideal
