"""机制5：FeedbackInjector 负反馈注入。

对应 DESIGN.md 2.4 节"机制 5"。
对应 Sea 的 docker-core/checkpoint/feedback_injector.go。

为什么需要：
    LLM 修复失败后，下一次重试时如果不告诉它"上次你试过 X 方案失败了"，
    它可能再次尝试同样的方案。FeedbackInjector 把失败记录拼成 system prompt
    注入，明确告诉 LLM "DO NOT try these approaches again"。
"""

from scholar_agent.models.harness import HarnessAttempt

MAX_RECENT = 3       # 只保留最近 3 次失败（避免 prompt 过长）
MAX_ERROR_CHARS = 500  # 每次失败截断 error 到 500 字符


def build_feedback_prompt(failed_attempts: list[HarnessAttempt]) -> str:
    """生成负反馈提示词，注入到 LLM 修复请求的 system prompt。

    空列表返回空字符串（首次修复没有历史可注入）。
    """
    if not failed_attempts:
        return ""

    recent = failed_attempts[-MAX_RECENT:]
    lines = ["[System Guard - Failed Approaches] DO NOT try these approaches again:"]
    for attempt in recent:
        error_preview = (attempt.error or "")[:MAX_ERROR_CHARS]
        lines.append(f"\n--- Attempt {attempt.attempt} (exit_code={attempt.exit_code}) ---")
        lines.append(error_preview)

    return "\n".join(lines)
