"""Phase 1 IR metrics — pure functions, no I/O, no side effects.

通用输入约定：
    ranked_doc_ids: list[str]        # 检索器返回的 doc_id，按相关性从高到低
    relevance:      dict[str, float] # ground-truth: doc_id -> graded relevance
                                     # ESCI 推荐: E=4, S=2, C=1, I=0
                                     # 二值指标 (Recall/MRR/MAP) 把 >0 视为相关
    k:              int              # cut-off

所有函数返回 float (0.0 ~ 1.0)。空召回 / 空 gold 的边界行为见各文件。
"""

from evals.metrics.map_score import average_precision, mean_average_precision
from evals.metrics.mrr import mrr_at_k, reciprocal_rank
from evals.metrics.ndcg import dcg_at_k, ndcg_at_k
from evals.metrics.recall import recall_at_k

__all__ = [
    "average_precision",
    "dcg_at_k",
    "mean_average_precision",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
