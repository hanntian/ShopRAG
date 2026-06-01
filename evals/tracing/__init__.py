"""Langfuse tracing 薄封装。

设计原则:
1. metrics 模块和 tracing 解耦 — metrics 只返回数字
2. 没装 langfuse / 没配 env 变量时静默 no-op，不影响离线 benchmark
3. 单向上报，不读 Langfuse 数据回来（保持纯粹）
"""

from evals.tracing.langfuse import (
    LangfuseTracer,
    flush,
    get_tracer,
    is_enabled,
)

__all__ = ["LangfuseTracer", "flush", "get_tracer", "is_enabled"]
