"""Harness 约束层：LLM 能力边界界定（⭐ 项目灵魂）。

设计哲学：模型负责建议，代码负责约束。
所有 LLM 产出都要被代码层校验——防造假、防越权、防死循环、防重复犯错。

5 个边界机制：
    1. patch_policy:        补丁静态校验（防 LLM 越权）
    2. fingerprint:         SHA256 + Repository Fingerprint（防代码偷换）
    3. metrics_recompute:   指标重算防伪（防 LLM 造假，⭐ Harness 灵魂）
    4. circuit_breaker:     CircuitBreaker（防 LLM 死循环）
    5. feedback_injector:   FeedbackInjector（防 LLM 重复犯错）

runner.py 把 5 个机制串到 run_with_healing 主循环里。

和 agents/ 的关系：
    agents/ 是"执行层"（4 个 Agent 干活）
    harness/ 是"约束层"（校验 LLM 产出，确保可信）
    ResearchCodingAgent 调 Harness.run_with_healing 跑代码 + 自愈

（各机制从子模块直接 import，包级无再导出——Middle Man 不留）
"""
