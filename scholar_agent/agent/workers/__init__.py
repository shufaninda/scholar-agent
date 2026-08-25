"""Agent 层：Librarian / Coder / ResearchCoding / Data。

workers 是"执行层"（4 个 Agent 干活），harness 是"约束层"（校验 LLM 产出）。
execute_step 按 Step.agent 路由到这里的 Agent。

（各 Agent 从子模块直接 import，包级无再导出——Middle Man 不留）
"""
