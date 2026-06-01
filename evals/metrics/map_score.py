"""MAP@K — Mean Average Precision.

单 query Average Precision (AP@K):
    AP@K = (1 / min(|gold|, K)) * sum_{i=1..K} P@i * rel_i
    rel_i 是 0/1（这里把 graded relevance > 0 视为 1）。

为什么分母用 min(|gold|, K)：当 |gold| > K 时再除以 |gold| 会人为压低
分数，TREC 习惯也是用 min。

MAP = 多 query AP 的均值（由 mean_average_precision 完成）。

文件名用 map_score.py 而非 map.py，避免和内置 `map` 撞名。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def average_precision(
    ranked_doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gold = {doc_id for doc_id, rel in relevance.items() if rel > 0}
    if not gold:
        return 0.0

    hits = 0
    score_sum = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in gold:
            hits += 1
            # P@(i+1) = hits / (i+1)
            score_sum += hits / (i + 1)

    denom = min(len(gold), k)
    return score_sum / denom


def mean_average_precision(
    runs: Iterable[tuple[Sequence[str], Mapping[str, float]]],
    k: int,
) -> float:
    runs_list = list(runs)
    if not runs_list:
        return 0.0
    total = sum(average_precision(ranked, rel, k) for ranked, rel in runs_list)
    return total / len(runs_list)
