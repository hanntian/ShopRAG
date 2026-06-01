"""Phase 1 metrics 单元测试。

测试策略：
1. 边界 case：空 gold、空召回、k=1
2. 已知答案 sanity check（手算过的小例子）
3. 关键不变性：完美排序 nDCG=1.0；全错 Recall=0
"""

from __future__ import annotations

import math

import pytest

from evals.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------- nDCG ----------


def test_ndcg_perfect_ranking_is_one():
    # ESCI-style: E=4, S=2, I=0
    relevance = {"a": 4, "b": 2, "c": 0}
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_reverse_ranking_below_one():
    relevance = {"a": 4, "b": 2}
    ranked = ["b", "a"]  # 高相关排到第二
    assert 0 < ndcg_at_k(ranked, relevance, k=2) < 1.0


def test_ndcg_empty_gold_is_zero():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_ndcg_no_hits_is_zero():
    relevance = {"x": 4}
    assert ndcg_at_k(["a", "b"], relevance, k=2) == 0.0


def test_ndcg_known_value():
    # 手算: rel = [3, 2, 0, 1], gain = 2^r - 1 = [7, 3, 0, 1]
    # discount = log2(i+2) for i=0..3 = [1, log2(3), 2, log2(5)]
    # DCG = 7/1 + 3/log2(3) + 0/2 + 1/log2(5)
    relevance = {"a": 3, "b": 2, "c": 0, "d": 1}
    ranked = ["a", "b", "c", "d"]
    expected_dcg = 7 / 1 + 3 / math.log2(3) + 0 / 2 + 1 / math.log2(5)
    # IDCG: 排序后 [3, 2, 1, 0] -> gain [7, 3, 1, 0]
    expected_idcg = 7 / 1 + 3 / math.log2(3) + 1 / 2 + 0
    assert ndcg_at_k(ranked, relevance, k=4) == pytest.approx(
        expected_dcg / expected_idcg
    )


def test_ndcg_linear_gain():
    relevance = {"a": 3, "b": 1}
    ranked = ["a", "b"]
    # linear: gain = rel
    # DCG = 3/1 + 1/log2(3); IDCG 同
    assert ndcg_at_k(ranked, relevance, k=2, gain="linear") == pytest.approx(1.0)


def test_dcg_invalid_k():
    with pytest.raises(ValueError):
        dcg_at_k(["a"], {"a": 1}, k=0)


def test_ndcg_invalid_gain():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1}, k=1, gain="bogus")


# ---------- Recall ----------


def test_recall_full_hit():
    relevance = {"a": 1, "b": 2}
    assert recall_at_k(["a", "b", "c"], relevance, k=3) == 1.0


def test_recall_half_hit():
    relevance = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x"], relevance, k=2) == 0.5


def test_recall_cutoff_excludes_late_hits():
    relevance = {"a": 1, "b": 1}
    # b 在 rank 5，k=2 召回不进去
    assert recall_at_k(["x", "a", "y", "z", "b"], relevance, k=2) == 0.5


def test_recall_empty_gold():
    assert recall_at_k(["a"], {}, k=1) == 0.0


def test_recall_graded_treated_as_binary():
    # I (rel=0) 不算 gold
    relevance = {"a": 0, "b": 4}
    assert recall_at_k(["a", "b"], relevance, k=2) == 1.0  # 只有 b 是 gold


# ---------- MRR ----------


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b"], {"a": 1}, k=2) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, k=5) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y"], {"a": 1}, k=2) == 0.0


def test_reciprocal_rank_cutoff():
    # hit 在 rank 3，k=2 不算
    assert reciprocal_rank(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_mrr_average_over_queries():
    runs = [
        (["a", "x"], {"a": 1}),  # RR = 1
        (["x", "a"], {"a": 1}),  # RR = 0.5
        (["x", "y"], {"a": 1}),  # RR = 0
    ]
    assert mrr_at_k(runs, k=2) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mrr_empty_runs():
    assert mrr_at_k([], k=10) == 0.0


# ---------- MAP ----------


def test_ap_perfect_ranking():
    # 2 gold 全部排前面
    relevance = {"a": 1, "b": 1}
    assert average_precision(["a", "b", "x"], relevance, k=3) == pytest.approx(1.0)


def test_ap_one_relevant_at_rank_2():
    # gold = {a}; ranked = [x, a]; P@2 = 1/2; AP = 0.5 / min(1, 2) = 0.5
    assert average_precision(["x", "a"], {"a": 1}, k=2) == pytest.approx(0.5)


def test_ap_two_hits_interleaved():
    # gold = {a, b}; ranked = [a, x, b]; hits at rank 1 (P=1) and rank 3 (P=2/3)
    # AP = (1 + 2/3) / min(2, 3) = (5/3) / 2 = 5/6
    relevance = {"a": 1, "b": 1}
    assert average_precision(["a", "x", "b"], relevance, k=3) == pytest.approx(5 / 6)


def test_ap_empty_gold():
    assert average_precision(["a"], {}, k=1) == 0.0


def test_map_average():
    runs = [
        (["a"], {"a": 1}),  # AP = 1
        (["x", "a"], {"a": 1}),  # AP = 0.5
    ]
    assert mean_average_precision(runs, k=2) == pytest.approx(0.75)
