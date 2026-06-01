"""Evaluation harness for DeepResearch-Stack.

Layout:
    metrics/   纯函数 IR 指标 (Recall@K, nDCG@K, MRR@K, MAP@K)
    runners/   把 metrics 和检索系统粘起来，跑 ESCI / WANDS 等 benchmark
    tracing/   Langfuse 薄封装，把每次实验上报到 dashboard
"""
