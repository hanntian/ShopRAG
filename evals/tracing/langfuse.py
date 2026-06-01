"""Langfuse 薄封装。

为什么不直接用 langfuse SDK：
1. 想要的接口正好就 3 个（start_trace / log_retrieval / log_score），SDK 表面太大
2. 没装 / 没配 env 时需要静默 no-op，runner 代码不用到处写 if
3. 屏蔽 v3 → v4 之类的 API 变化，集中改一处

环境变量（Cloud 免费档）:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://us.cloud.langfuse.com   # 或 https://cloud.langfuse.com (EU)

用法:

    from evals.tracing import get_tracer

    tracer = get_tracer()
    with tracer.trace(name="bm25-baseline", input={"query": q}) as trace:
        ranked = retriever.search(q, k=100)
        trace.log_retrieval(ranked, k=10)
        trace.log_score("ndcg@10", ndcg_at_k(ranked, gold, 10))
        trace.log_score("recall@100", recall_at_k(ranked, gold, 100))

    tracer.flush()   # 脚本结束时调一次
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    """检查是否同时满足：langfuse 已安装 + env 变量已配置。"""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


class _NoopTrace:
    """没启用时返回这个，方法全是 no-op，调用方不用判断。"""

    trace_id: str = ""

    def log_retrieval(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_score(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class _ActiveTrace:
    """包装一个 Langfuse span，提供 log_retrieval / log_score。"""

    def __init__(self, client: Any, span: Any) -> None:
        self._client = client
        self._span = span
        self.trace_id: str = getattr(span, "trace_id", "")

    def log_retrieval(
        self,
        ranked_doc_ids: Sequence[str],
        *,
        k: int | None = None,
        scores: Sequence[float] | None = None,
        name: str = "retrieval",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """把检索结果作为子 observation (as_type='retriever') 挂上去。

        显式存 ranked_doc_ids + scores 而不是塞进 metadata 字符串，
        方便后续在 Langfuse UI 里直接看。as_type='retriever' 会在 UI 里
        用专用图标，和普通 span 区分开。
        """
        topk = list(ranked_doc_ids[:k]) if k else list(ranked_doc_ids)
        output: dict[str, Any] = {"ranked_doc_ids": topk}
        if scores is not None:
            output["scores"] = list(scores[: len(topk)])
        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type="retriever",
                output=output,
                metadata=dict(metadata) if metadata else None,
            ):
                pass
        except Exception:  # pragma: no cover - 上报失败不应中断 eval
            logger.exception("langfuse log_retrieval failed")

    def log_score(
        self,
        name: str,
        value: float,
        *,
        comment: str | None = None,
    ) -> None:
        """挂一个 numeric score 到当前 trace。

        ScoresTable / ChartScores 会自动按 name 聚合并出图。
        """
        if not self.trace_id:
            return
        try:
            # v4 SDK: create_score (numeric). 必须是 kwargs only.
            self._client.create_score(
                trace_id=self.trace_id,
                name=name,
                value=float(value),
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception:  # pragma: no cover
            logger.exception("langfuse log_score failed: name=%s", name)

    def update(self, **kwargs: Any) -> None:
        """转发到底层 span.update（input / output / metadata 等）。"""
        try:
            self._span.update(**kwargs)
        except Exception:  # pragma: no cover
            logger.exception("langfuse span.update failed")


class LangfuseTracer:
    """单例风格的 tracer。没配 env / 没装 langfuse -> 整体 no-op。

    线程安全留给底层 SDK 处理；这里只做包装。
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        self.enabled: bool = is_enabled()
        if self.enabled:
            try:
                from langfuse import Langfuse

                # Langfuse() 会自动读 env：PUBLIC/SECRET/HOST
                self._client = Langfuse()
            except Exception:
                logger.exception(
                    "Failed to init Langfuse client; tracing disabled"
                )
                self.enabled = False

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[_ActiveTrace | _NoopTrace]:
        """开一个 trace（作为 root span）。

        Args:
            name: span 名字，e.g. "bm25-baseline-q-42"
            input: 输入字典（一般放 query）
            metadata: 附加元数据（dataset、model、git_sha…）
        """
        if not self.enabled or self._client is None:
            yield _NoopTrace()
            return

        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type="span",
                input=dict(input) if input else None,
                metadata=dict(metadata) if metadata else None,
            ) as span:
                yield _ActiveTrace(self._client, span)
        except Exception:  # pragma: no cover
            logger.exception("langfuse trace failed; falling back to no-op")
            yield _NoopTrace()

    def flush(self) -> None:
        """脚本结束时调一次，确保所有事件都上报。"""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:  # pragma: no cover
            logger.exception("langfuse flush failed")


# 全局单例 — runner 直接 get_tracer() 拿同一个
_singleton: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    global _singleton
    if _singleton is None:
        _singleton = LangfuseTracer()
    return _singleton


def flush() -> None:
    """模块级快捷方式。"""
    get_tracer().flush()
