"""Recall@K.

含义：top-K 里命中了多少 gold 相关文档 / gold 相关文档总数。

graded relevance 处理：任何 relevance > 0 的 doc_id 都视为"相关"。
对 ESCI 来说就是 E/S/C 都算，只有 I 不算。如果只想算 E 命中，调用方在
传入前自己过滤 relevance 字典即可。

边界：
    - gold 为空 -> 0.0（避免 0/0）
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gold = {doc_id for doc_id, rel in relevance.items() if rel > 0}
    if not gold:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(top_k & gold) / len(gold)
