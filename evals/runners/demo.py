"""最小 demo runner — 验证 metrics + Langfuse 端到端。

用一份硬编码的 toy 数据集（3 个 query，5 个 doc），跑一个 "假" 的检索器，
计算 Phase 1 全部指标并上报 Langfuse。

用途:
1. 没启动 Qdrant、没下 ESCI 时也能跑通整条 pipeline
2. CI 烟测：metrics + tracing 任何一边坏了立刻挂
3. Langfuse dashboard 第一次填数据，确认 ScoresTable / ChartScores 工作

跑法:
    # 不上报，纯打印
    uv run python -m evals.runners.demo

    # 上报到 Langfuse Cloud (免费档)
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://us.cloud.langfuse.com
    uv run python -m evals.runners.demo
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evals.metrics import (
    mean_average_precision,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from evals.tracing import get_tracer


@dataclass
class EvalQuery:
    qid: str
    query: str
    # gold relevance: doc_id -> graded label (ESCI: E=4, S=2, C=1, I=0)
    relevance: Mapping[str, float]


def _toy_dataset() -> list[EvalQuery]:
    """3 个 query, ESCI 风格四级标签。"""
    return [
        EvalQuery(
            qid="q1",
            query="vLLM paged attention",
            relevance={"d1": 4, "d2": 2, "d3": 0, "d4": 1, "d5": 0},
        ),
        EvalQuery(
            qid="q2",
            query="rotary positional embedding",
            relevance={"d2": 4, "d5": 2, "d1": 0},
        ),
        EvalQuery(
            qid="q3",
            query="speculative decoding throughput",
            relevance={"d4": 4, "d3": 2, "d1": 1},
        ),
    ]


def _fake_retriever(query: str, k: int = 5) -> tuple[Sequence[str], Sequence[float]]:
    """假装是个检索器：根据 query 关键字返回固定排序。

    真实场景这里换成 HybridRetriever / BM25 / Dense 即可。
    """
    if "vllm" in query.lower():
        return ["d1", "d2", "d4", "d3", "d5"][:k], [0.9, 0.7, 0.5, 0.3, 0.1][:k]
    if "rotary" in query.lower():
        return ["d2", "d1", "d5", "d3", "d4"][:k], [0.8, 0.6, 0.5, 0.2, 0.1][:k]
    return ["d4", "d3", "d1", "d2", "d5"][:k], [0.9, 0.6, 0.4, 0.2, 0.1][:k]


def run(retriever_name: str = "fake-retriever") -> dict[str, float]:
    """跑 demo eval，返回汇总指标 dict。"""
    dataset = _toy_dataset()
    tracer = get_tracer()
    print(f"[demo] langfuse enabled: {tracer.enabled}")

    # 收集 (ranked, relevance) 给 mrr/map 求平均
    runs: list[tuple[Sequence[str], Mapping[str, float]]] = []
    per_query_ndcg10: list[float] = []
    per_query_recall100: list[float] = []

    for ex in dataset:
        ranked, scores = _fake_retriever(ex.query, k=5)
        runs.append((ranked, ex.relevance))

        ndcg10 = ndcg_at_k(ranked, ex.relevance, k=10)
        recall100 = recall_at_k(ranked, ex.relevance, k=100)
        per_query_ndcg10.append(ndcg10)
        per_query_recall100.append(recall100)

        # 每条 query 开一个 trace
        with tracer.trace(
            name=f"{retriever_name}-{ex.qid}",
            input={"query": ex.query},
            metadata={
                "qid": ex.qid,
                "retriever": retriever_name,
                "dataset": "toy",
            },
        ) as trace:
            trace.log_retrieval(ranked, scores=scores, k=5)
            trace.log_score("ndcg@10", ndcg10)
            trace.log_score("recall@100", recall100)
            trace.update(output={"top_doc": ranked[0] if ranked else None})

    # 汇总
    summary = {
        "ndcg@10": sum(per_query_ndcg10) / len(per_query_ndcg10),
        "recall@100": sum(per_query_recall100) / len(per_query_recall100),
        "mrr@10": mrr_at_k(runs, k=10),
        "map@10": mean_average_precision(runs, k=10),
    }

    # 汇总也开一个 trace（可选）
    with tracer.trace(
        name=f"{retriever_name}-summary",
        metadata={"retriever": retriever_name, "dataset": "toy", "n_queries": len(dataset)},
    ) as trace:
        for name, val in summary.items():
            trace.log_score(name, val, comment="dataset mean")

    print("\n[demo] summary:")
    for name, val in summary.items():
        print(f"  {name:<12} = {val:.4f}")

    tracer.flush()
    return summary


if __name__ == "__main__":
    run()
