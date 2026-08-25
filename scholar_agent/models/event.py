"""PlanEvent 模型：SSE 推送给前端的事件。

为什么需要：
    前端通过 SSE 实时展示任务进度 + 沙箱输出。每个事件对应一个 SSE 消息。
    事件经 event_sink 回调直推 SSE（旁路），不进 AgentState。
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlanEventType(str, Enum):
    """事件类型枚举。

    前端按 type 决定怎么渲染（如 sandbox_output 追加到控制台，task_completed 标绿）。
    """

    task_ready = "task_ready"            # 任务就绪（依赖完成）
    task_started = "task_started"        # 任务开始执行
    task_completed = "task_completed"    # 任务成功完成
    task_failed = "task_failed"          # 任务失败
    task_blocked = "task_blocked"        # 任务被阻塞（上游失败）
    task_retrying = "task_retrying"      # 任务重试中
    artifact_created = "artifact_created"  # 产出新 artifact
    plan_awaiting_approval = "plan_awaiting_approval"  # 计划等待人工审批（图挂起中）
    plan_completed = "plan_completed"    # 整个 plan 完成
    plan_failed = "plan_failed"          # 整个 plan 失败
    plan_canceled = "plan_canceled"      # 整个 plan 被取消/放弃（用户主动终止）
    sandbox_output = "sandbox_output"    # 沙箱实时输出（流式）


class PlanEvent(BaseModel):
    """SSE 推送的事件。

    graph 节点经 _emit → event_sink 回调直推 SSE（api/sse.py），
    不进 state。每个事件对应一个 SSE 消息：event: {type}\\ndata: {json}\\n\\n
    """

    type: PlanEventType
    """事件类型。"""

    task_id: str | None = None
    """关联的 Step ID（sandbox_output / task_* 事件有值）。"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """事件发生时间。"""

    data: dict[str, Any] = Field(default_factory=dict)
    """附加数据。如 {"stdout": "...", "stderr": "..."} 或 {"artifact_key": "..."}。"""

    model_config = {"protected_namespaces": ()}
