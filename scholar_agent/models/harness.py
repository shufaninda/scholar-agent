"""Harness 数据模型：自愈执行报告 + 每次尝试记录。

为什么需要：
    Harness 跑完后要返回结构化报告——记录每次 attempt 的退出码/错误/是否修复过，
    以及最终状态（passed/failed）和原因。前端展示 + 调试都依赖这个。

字段一致性说明（解决 P1-14）：
    code_hash 必须有默认值 None，因为首次 attempt 不需要 hash（只有 attempt > 1
    才需要对比 hash 检测代码是否变了）。
"""

from pydantic import BaseModel, Field


class HarnessAttempt(BaseModel):
    """单次修复尝试的记录。

    每次 attempt（无论成功失败）都会创建一个 HarnessAttempt 加入 attempts 列表。
    """

    attempt: int
    """第几次尝试（1-based）。"""

    exit_code: int
    """退出码。0=成功，非 0=失败，-1=未执行（Patch Policy 拒绝 / 代码未变化）。"""

    error: str = ""
    """错误信息（stderr 或拒绝原因）。截断到合理长度避免 prompt 过长。"""

    repaired: bool = False
    """是否经过 LLM 修复。首次 attempt=False，后续 attempt=True。"""

    code_hash: str | None = None
    """本次执行时代码的 SHA256。
    ⭐ 用于检测 LLM 修复后代码是否真的变了（机制2）。
    首次 attempt 可以为 None（不需要对比）。"""


class HarnessReport(BaseModel):
    """Harness 自愈执行的整体报告。

    无论成功失败都返回这个，调用方（ResearchCodingAgent）根据 status 判断。
    """

    status: str
    """最终状态："passed" 或 "failed"。"""

    attempts: list[HarnessAttempt] = Field(default_factory=list)
    """每次 attempt 的记录列表。"""

    final_result: str = ""
    """成功时的最终输出（stdout）。"""

    reason: str = ""
    """失败原因。如 "failed after 3 attempts" 或 CircuitBreaker 熔断原因。"""

    model_config = {"protected_namespaces": ()}
