"""Plan 模型：一次请求的完整执行计划（步骤列表 + 产物仓库）。

替代原 PlanGraph。瘦身的字段及理由：
- edges: list[TaskEdge] —— 边即数组下标，无需独立表达；
- budget: RunBudget —— 被"单步 3 次重试 + fail-fast"替代；
- meta: GraphMeta —— 前端进度直接由 steps 状态推导，不再单独存。

保留的两大核心：
- steps：LLM 规划、validator 校验后的步骤数组（控制面）；
- artifacts：Agent 间交接产物的全局仓库（数据面）。
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from scholar_agent.models.artifact import Artifact
from scholar_agent.models.step import Step


class PlanStatus(str, Enum):
    """Plan 的生命周期状态（审批 + 执行 + 终态）。"""

    pending = "pending"            # 已规划，等待审批/执行
    in_progress = "in_progress"    # 执行中
    completed = "completed"        # 全部步骤成功
    failed = "failed"              # 有步骤终局失败
    canceled = "canceled"          # 用户取消/放弃


class Plan(BaseModel):
    """完整执行计划：steps（做什么）+ artifacts（已产出什么）。

    planner_node 产出，execute_step 节点消费，report_node 收尾。
    全程住在 state["plan"] 里随 checkpoint 持久化——thread_id = plan.id
    即断点续跑的寻址钥匙。
    """

    id: str
    """计划唯一 ID（= checkpoint 的 thread_id）。"""

    trace_id: str = ""
    """链路追踪 ID，日志关联用。"""

    user_intent: str = ""
    """用户原始意图（rewritten 优先）。"""

    intent_type: str = ""
    """意图类型字符串（IntentType.value）。"""

    status: PlanStatus = PlanStatus.pending
    """计划状态。"""

    steps: list[Step] = Field(default_factory=list)
    """⭐ 有序步骤数组：下标即依赖即执行序。"""

    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    """⭐ 全局产物仓库（Agent 间消息总线）：完成时写入，执行前读取。"""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
