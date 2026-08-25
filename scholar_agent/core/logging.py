"""结构化日志：用 structlog 输出事件式日志。

对应 Sea 的 internal/logging/。
"""

import structlog

logger = structlog.get_logger()
