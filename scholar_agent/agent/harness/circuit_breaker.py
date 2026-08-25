"""机制4：CircuitBreaker 防死循环。

为什么需要：
    LLM 修复代码时可能"鬼打墙"——多次重试都尝试同一个错误方案（生成同样的
    cmd+output）。CircuitBreaker 对 cmd+output 做 SHA256，滑动窗口内全相同
    就熔断，提前 fail-fast 不浪费 attempt。
"""

import hashlib
from collections import deque


class CircuitBreaker:
    """语义断路器：检测 LLM 是否陷入"同错误反复试"死循环。

    滑动窗口（Harness 用 window=2）：窗口填满且全相同才熔断，避免单次重复误判。
    只在失败信号（error:/failed/traceback/exception）下记录，避免对成功命令误判。
    """

    FAILURE_SIGNALS = ["error:", "failed", "traceback", "exception", "exit code"]

    def __init__(self, window: int = 2):
        """初始化断路器。

        Args:
            window: 滑动窗口大小。window=2 表示连续 2 次相同就熔断。
                    必须 ≤ Harness.max_attempts（默认 3）。
        """
        self.window = window
        self.hashes: deque[str] = deque(maxlen=window)

    def _is_failure(self, output: str) -> bool:
        """判断输出是否为失败信号（大小写不敏感）。"""
        lowered = (output or "").lower()
        return any(sig in lowered for sig in self.FAILURE_SIGNALS)

    def should_break(self, cmd: str, output: str) -> tuple[bool, str]:
        """检查是否应该熔断。

        返回 (是否熔断, 原因)。熔断时原因含"建议人工介入"。

        - 不是失败信号 → 不记录，返回 (False, "")
        - 是失败信号 → 算 SHA256，加入 deque
        - deque 填满且全相同 → 返回 (True, 原因)
        """
        if not self._is_failure(output):
            return False, ""  # 成功命令不记录

        digest = hashlib.sha256(f"{cmd}\n{output}".encode()).hexdigest()
        self.hashes.append(digest)

        if len(self.hashes) == self.window and len(set(self.hashes)) == 1:
            return True, (
                f"CircuitBreaker 熔断：连续 {self.window} 次失败且输出完全相同，"
                f"疑似 LLM 陷入死循环，建议人工介入"
            )
        return False, ""
