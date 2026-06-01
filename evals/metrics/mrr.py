"""MRR@K — Mean Reciprocal Rank.

单 query 的 reciprocal rank = 1 / (第一个相关文档的 rank)，没命中就是 0。
MRR = 多 query 的均值（这里只算单 query 那一项，平均交给 runner）。

适用：当只关心"用户第一眼能不能看到相关结果"，比如导航式查询、首页推荐。
对 ESCI 这种多相关文档的场景，MRR 信号弱于 nDCG，但实现/解释都简单。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def reciprocal_rank(
    ranked_doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if relevance.get(doc_id, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def mrr_at_k(
    runs: Iterable[tuple[Sequence[str], Mapping[str, float]]],
    k: int,
) -> float:
    """对多个 query 求 MRR@K 平均。

    Args:
        runs: 可迭代 (ranked_doc_ids, relevance) 二元组，每个对应一个 query。
        k: 截断位置。
    """
    runs_list = list(runs)
    if not runs_list:
        return 0.0
    total = sum(reciprocal_rank(ranked, rel, k) for ranked, rel in runs_list)
    return total / len(runs_list)
