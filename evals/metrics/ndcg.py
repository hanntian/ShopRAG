"""nDCG@K — Normalized Discounted Cumulative Gain.

为什么 ESCI 要用 nDCG：
    ESCI 有四级标签 E(Exact)/S(Substitute)/C(Complement)/I(Irrelevant)，
    nDCG 是 graded relevance 的标准指标，能区分"把 Exact 排第 1"和
    "把 Substitute 排第 1"的差距，而 Recall/MRR 看不出来。

公式（Burges, 用得最广的一版）:
    gain_i     = 2^rel_i - 1
    discount_i = log2(i + 2)          # i 从 0 开始，第 1 名 discount=1
    DCG@K      = sum_{i=0..K-1} gain_i / discount_i
    nDCG@K     = DCG@K / IDCG@K       # IDCG = 把 gold 按相关性降序排出来的 DCG

边界：
    - 召回里没有 gold 命中 -> 0.0
    - gold 全空 -> 0.0（无可评估对象，不抛异常，方便聚合时直接平均）
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
    *,
    gain: str = "exp",
) -> float:
    """计算 DCG@K（未归一化）。

    Args:
        ranked_doc_ids: 检索结果，按预测相关性降序。
        relevance: doc_id -> graded relevance（缺失视为 0）。
        k: 截断位置，必须 >= 1。
        gain: "exp" 用 2^rel - 1（默认，Burges 版），"linear" 用 rel 本身。
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if gain not in {"exp", "linear"}:
        raise ValueError(f"gain must be 'exp' or 'linear', got {gain!r}")

    score = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = float(relevance.get(doc_id, 0.0))
        if rel <= 0:
            continue
        g = (2.0**rel - 1.0) if gain == "exp" else rel
        # i 从 0 开始：rank 1 的 discount 应为 log2(2) = 1
        score += g / math.log2(i + 2)
    return score


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
    *,
    gain: str = "exp",
) -> float:
    """nDCG@K = DCG@K / IDCG@K。

    IDCG@K 由 relevance 字典里所有非零项按降序排出，截到 K。
    """
    dcg = dcg_at_k(ranked_doc_ids, relevance, k, gain=gain)
    ideal_rels = sorted((r for r in relevance.values() if r > 0), reverse=True)
    if not ideal_rels:
        return 0.0
    # 用 placeholder doc_id "_ideal_{i}" 复用 dcg_at_k
    ideal_ids = [f"_ideal_{i}" for i in range(len(ideal_rels))]
    ideal_rel_map = dict(zip(ideal_ids, ideal_rels))
    idcg = dcg_at_k(ideal_ids, ideal_rel_map, k, gain=gain)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
