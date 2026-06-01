"""Eval runners.

把 metrics + tracing + 检索系统粘起来，跑一个完整 benchmark。
每个 runner 输出 (1) 终端汇总 (2) Langfuse trace + score（如果启用）。
"""
