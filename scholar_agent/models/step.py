"""Step 模型：执行计划中的一个步骤。

替代原 TaskNode + TaskEdge 的组合。设计动机（详见 graph.py 模块注释）：
论文复现的任务依赖恒为"前一步"，业务图 100% 是链式——数组下标同时
承载拓扑序、依赖序、执行序（四序合一），无需独立的 dependencies /
TaskEdge / 拓扑排序表达。

LLM 边界（"代码约束 LLM"在数据模型层的落地）：
    - LLM 只能填"设计字段"：name / type / agent / artifacts 契约，
      且必须过 validator 两道闸才能成为 Step；
    - status / result / error 是运行时状态，全部由代码（execute_step
      节点）填写，LLM 全程摸不到。
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """步骤生命周期（8 态瘦身为 5 态）。

    砍掉的三态及理由：
    - ready：链式下"依赖满足"就是"轮到它"，与 pending 无异；
    - blocked：连坐机制已删（fail-fast 替代——某步终局失败整图收尾）；
    - skipped：从未有真实跳过语义。

    状态流转：
        pending → in_progress → completed（正常）
                          ↓
                       failed（3 次尝试全败，fail-fast）
        pending → canceled（用户取消，剩余步骤由 execute_step 标记）
    """

    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class Step(BaseModel):
    """执行计划中的一个步骤：做什么、谁来做、领什么料、交什么货。

    执行顺序由所在 Plan.steps 数组的下标隐式决定（step[i] 依赖
    step[i-1] 的产物）——这就是链式 DAG 的最简等价表达。
    """

    id: str
    """步骤唯一 ID，如 "s1"。SSE 事件 task_id 用它归位。"""

    name: str
    """人类可读名字，如"解析论文"。前端展示 + 审批清单渲染用。"""

    type: str
    """步骤类型：paper_parse / repo_discovery / code_generate /
    code_run / framework_compare / report。Agent 内部按它二次分发。"""

    description: str = ""
    """步骤描述，给 LLM 看的 prompt 素材。"""

    agent: str
    """⭐ 派工钥匙：从 workers 名册里查这个字段路由到对应 Agent。
    只能填注册名（librarian_agent 等），validator 高契约闸把关。"""

    status: StepStatus = StepStatus.pending
    """运行时状态，初始 pending。只由代码填写。"""

    # ─── 产物契约（validator 两道闸的校验标的）───

    required_artifacts: list[str] = Field(default_factory=list)
    """需要消费的上游产物名。⭐ 校验：必须有 producer。"""

    output_artifacts: list[str] = Field(default_factory=list)
    """本步骤产出的产物名。⭐ 校验：跨步骤不能重复（写冲突检测）。"""

    # ─── 执行参数 / 结果（代码填，LLM 不碰）───

    inputs: dict[str, object] = Field(default_factory=dict)
    """执行参数，如 {"paper_title": "Attention Is All You Need"}。
    planner 从意图 entities 提取填充（确定性代码，不走 LLM）。"""

    timeout_seconds: int = 600
    """单次执行超时（沙箱 kill 容器用）。"""

    result: str | None = None
    """执行成功后记一笔"交了哪些货"（artifact key 列表）。"""

    error: str | None = None
    """执行失败的错误信息（截断至 500 字符）。"""

    # ─── 时间戳（前端展示用）───

    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
