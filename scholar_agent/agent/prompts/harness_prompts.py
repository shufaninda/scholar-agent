"""Harness 代码修复提示词。

对应 Sea 的 RuntimeCodeRepairUserPrompt。
对应 DESIGN.md 2.4 节"机制 5：FeedbackInjector"。

Harness 的 _repair_code 用这个提示词调 LLM 修复代码：
    传入：原代码 + traceback（+ 历史失败负反馈）
    输出：修复后的完整代码

复刻自 Sea 的 prompts.go 395-428 行。
"""

# ──────────────────────────────────────────────
# 代码修复 system prompt
# ──────────────────────────────────────────────

REPAIR_SYSTEM = """你是一个 Python 代码修复助手。你的任务是根据错误日志修复代码，返回修复后的完整代码。

要求：
1. 优先修复第三方库 API/导入路径兼容问题，例如库升级后类或函数被迁移。
2. 如果是 SyntaxError 或 f-string 语法错误，必须修正为合法的 Python 3.12 语法。
3. 不要在代码里增加 pip install 之类的安装语句（依赖恢复走单独的流程）。
4. 不要把论文模型、embedding、LLM、检索器或核心算法替换为 mock/fake 实现。
5. 如果错误涉及 API Key 或外部服务凭证缺失，应返回能清晰报告 unavailable 的代码，不要伪造结果。
6. 只返回修复后的完整 Python 代码，不要 markdown，不要解释。
"""


# ──────────────────────────────────────────────
# 代码修复 user prompt 拼接
# ──────────────────────────────────────────────

def repair_user_prompt(code: str, error: str) -> str:
    """代码修复的 user prompt：传入原代码 + 错误日志。

    Args:
        code: 运行失败的原代码
        error: stderr / traceback

    Returns:
        拼接好的 user prompt
    """
    return (
        f"下面这段 Python 代码运行失败，请根据错误日志直接修复完整代码。\n\n"
        f"错误日志：\n{error}\n\n"
        f"原始代码：\n```python\n{code}\n```"
    )
