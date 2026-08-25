"""LangGraph 全局状态定义。

字段两组：
    1. 主流程数据：user_input / session_id / plan_id / intent / plan /
       artifacts / final_report / error
    2. 执行游标：current_step（走到第几步）+ revision_count（审批修订计数）

原"执行循环账本"（current_task / attempt_counts / total_attempts /
feedback 四字段 + merge_counts reducer）已删——被 execute_step 节点的
"单步 3 次原地重试 + fail-fast"替代，调度层不再需要知道"重试了几次、
被谁连坐、预算还剩多少"这些中间态。

注意：没有 events 字段——事件走 event_sink 回调直推 SSE（旁路），
不经过 state。

⭐ Reducer（Annotated 第二参数）——字段的"入账规则"：
    无 reducer 的字段：节点返回值 = 新值，直接覆盖。
    有 reducer 的字段：节点返回值 = 增量，框架执行 reducer(旧值, 增量)。
    revision_count 用 operator.add：approval 的 revise 分支返回 1，
    框架累加；/resume 端点读它做封顶判定（3 次上限，防无限重规划）。
"""

import operator
from typing import Annotated, TypedDict

from scholar_agent.models.artifact import Artifact
from scholar_agent.models.intent import IntentContext
from scholar_agent.models.plan import Plan


class AgentState(TypedDict):
    """LangGraph 全局状态，在节点间自动传递。

    每个节点 return 后框架配合 Checkpointer 自动持久化
    （thread_id = plan_id，断点续跑的钥匙）。
    """

    user_input: str
    """用户原始输入。"""

    session_id: str
    """会话 ID（意图记忆用）。/runs 入口传入。"""

    plan_id: str
    """API 层预生成的 plan ID。

    /runs 一段式入口用：SSE 队列必须先于图执行订阅（防丢事件），
    所以 ID 由 API 层生成传进来，planner_node 用它覆写 plan.id。
    """

    intent: IntentContext
    """意图识别结果（intent_node 产出）。"""

    plan: Plan
    """执行计划（planner_node 产出；图级直连入口直接传入现成的）。"""

    artifacts: dict[str, Artifact]
    """Agent 间产物流转容器（execute_step 成功后写回，与 plan 同步）。"""

    final_report: str
    """最终报告（report_node 产出）。"""

    error: str | None
    """错误信息，任一环节终局失败时填充。"""

    # ── 执行游标 ──

    current_step: int
    """⭐ 执行到第几步（0-based 游标）。

    execute_step 自环节点唯一推进（成功后 +1 随 checkpoint 落盘），
    LLM 全程摸不到。恢复时从 checkpoint 读回，跳过已完成的步骤——
    断点续跑粒度 = 步。
    """

    revision_count: Annotated[int, operator.add]
    """用户 revise（带新输入重新规划）的累计次数。

    上限拦在 API 层（MAX_REVISIONS=3），图内只如实记账——职责分工。
    """
