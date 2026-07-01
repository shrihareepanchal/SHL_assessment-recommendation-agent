
from __future__ import annotations


def recall_at_k(recommended_urls: list[str], relevant_urls: set[str], k: int = 10) -> float:
    if not relevant_urls:
        return 0.0
    top_k = set(recommended_urls[:k])
    hit = len(top_k & relevant_urls)
    return hit / len(relevant_urls)


def mean_recall_at_k(per_trace_recalls: list[float]) -> float:
    if not per_trace_recalls:
        return 0.0
    return sum(per_trace_recalls) / len(per_trace_recalls)
