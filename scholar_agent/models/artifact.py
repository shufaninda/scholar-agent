"""Artifact 模型：Agent 之间传递的产物。

对应 Sea 的 Artifact struct（artifact.go）。
对应 DESIGN.md 0 节 State 定义里的 artifacts: dict[str, Artifact]。

为什么需要：
    Agent 间产物流转的核心载体。LibrarianAgent 产出 parsed_paper，
    CoderAgent 消费 parsed_paper 产出 generated_code，DataAgent 消费所有 artifact
    产出 final_report。Plan.artifacts 是全局容器，每个 Agent 完成后写入。

【AI 生成】类骨架（你审字段一致性）
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Artifact 类型枚举。

    对应 Sea 的 inferArtifactType 关键词推断。
    executor 根据 artifact 内容关键词推断类型，存入 type 字段。
    下游消费方可以按类型决定怎么处理（如 image_base64 转图片展示）。
    """

    json = "json"                    # JSON 数据
    image_base64 = "image_base64"    # base64 编码的图片
    code = "code"                    # 代码文本
    url = "url"                      # URL 字符串
    text = "text"                    # 普通文本
    metrics = "metrics"              # 指标数据（dict）
    report = "report"                # 报告文本
    dependency_spec = "dependency_spec"  # 依赖规格


class Artifact(BaseModel):
    """Agent 间传递的产物。

    生产：Agent 执行完成后，把 output_artifacts 对应的 Artifact 写入 Plan.artifacts。
    消费：下一个 Agent 执行前，从 Plan.artifacts 按 required_artifacts 取值。
    """

    key: str
    """artifact 唯一键，如 "parsed_paper" / "generated_code" / "final_report"。
    必须和 Step.output_artifacts / required_artifacts 里的名字对齐。"""

    type: ArtifactType = ArtifactType.text
    """artifact 类型，下游按类型决定怎么处理。"""

    producer_task_id: str = ""
    """产出这个 artifact 的 Step ID。调试溯源用。"""

    value: Any = None
    """artifact 的值。可能是 str / dict / list / bytes 等，按 type 解释。"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """附加元信息，如 {"model": "deepseek-chat", "tokens": 1234}。"""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
