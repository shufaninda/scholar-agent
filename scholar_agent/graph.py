"""LangGraph StateGraph 构建：Plan-Execute 多 Agent 主流程（步骤自环版）。

这是整个项目的编排核心——基于 LangGraph 的调度编排。

主图（静态，编译期固定 5 个节点）：

    intent ──Command──► planner ──► approval ──approve──► execute_step ──┐
      │ （plan 已存在：图级直连入口）  ▲（interrupt 挂起）  ▲            │（还有步骤）
      │  └──Command──► execute_step   │（revise：带新输入   │            │（自环：游标+1）
      │                               │  回炉重新规划）      │            │
      └─（意图不合格）─Command──► END  └─ abandon ─► report ◄────────────┘
                                                    ▲（走完/失败/取消）
                                                report ──► END

⭐ 核心设计——步骤列表替代 DAG 调度：
    论文复现的任务依赖恒为"前一步"（解析→搜repo→跑代码→报告），
    业务图 100% 是链式。链是 DAG 的特例：数组下标同时承载拓扑序、
    依赖序、执行序（四序合一），独立的 dependencies / TaskEdge /
    拓扑排序 / 分层并行全部冗余。原 pick_task→worker→verify 三节点
    循环（挑单/执行/三本账裁决，约 300 行）退化为 execute_step 单
    节点自环 + current_step 游标（约 60 行）。

⭐ 多 agent 不变（与调度结构无关）：
    4 个异构 Agent（librarian 纯 LLM / coder LLM+沙箱 / research
    确定性工具链+Harness / data 纯 LLM）+ LLM 动态编排步骤（选谁、
    几步、什么序，每次请求现编）+ artifacts 全局仓库做 Agent 间
    消息总线。架构即 LangGraph 官方 plan-and-execute 模式。

⭐ 代码约束 LLM——LLM 在执行层的触点为零：
    - 出什么步骤：LLM 提议，validator 两道闸验收（产物契约闭合 +
      高契约步骤派工），不过则模板兜底；
    - 先跑哪步：数组下标（数学性质，不是 LLM"觉得"）；
    - 派给谁：workers 名册查表（agent 字段只能填注册名）；
    - 要不要重试：MAX_STEP_ATTEMPTS 计数器（确定性代码）；
    - 代码挂了怎么修：LLM 提议，Harness 5 道闸验收（worker 墙内）；
    - 跑没跑成：exit_code / 指纹 / 指标重算（确定性代码）。

⭐ 失败处理——单步小重试 + fail-fast：
    每步失败原地重试 3 次（防偶发网络/超时抖动，防必死任务烧资源）；
    3 次全败判 failed 直接进 report（fail-fast）——链式下下游全部
    依赖本步，跑下去也必然失败，不烧冤枉钱。原 verify 的三本账
    （attempt_counts / retry_limit / budget）+ 连坐闭包整体退役。

⭐ 人工审批——LangGraph 原生 interrupt()：
    approval 节点在 planner 之后调用 interrupt() 挂起整张图，审批
    请求（plan 全量 + 已修订次数）作为 interrupt payload 随
    checkpoint 持久化。前端经 SSE 收到 plan_awaiting_approval，
    用户决定后调 POST /plans/{id}/resume 以 Command(resume=决定) 唤醒。
    approval 节点体只有 interrupt() 一件事：resume 会从挂起节点
    第一行重跑，interrupt 之前不能有任何副作用（不调 LLM、不发
    事件、不改 state），否则恢复时全部重复执行。

⭐ 导航机制——Command API：
    需要动态分流的节点（intent / approval / execute_step）直接
    return Command(goto=..., update=...)——去向和 state 更新在同一
    个 return 里原子表达。固定路径（planner→approval、report→END）
    仍用静态边。返回 Command 的节点一律不挂静态出边。

⭐ 断点续跑粒度 = 步：
    execute_step 每完成一步 return 一次，plan（含步骤状态 + 产物
    仓库）+ current_step 游标随 checkpoint 原子落盘。崩溃后恢复，
    已完成的步骤不重跑。游标唯一推进点在成功 return 里——落盘与
    完成同一瞬间，游标不可能停在"做了一半"的位置。

⭐ 容灾分工（决策跟图走，能力跟模块走）：
    - 单步自愈（偶发抖动重跑 / 代码挂了 LLM 修）在 execute_step
      与 worker 墙内；
    - 跨步骤止损（fail-fast 收尾）在 execute_step 出口；
    - 7 个降级策略原地不动（classifier/planner/harness/依赖恢复/
      clone 降级/记忆降级/沙箱两阶段），图节点只调用不搬运。

依赖注入：build_graph(llm, sandbox, ...) 组装 workers + classifier，
测试时传 agents 替身即可，不起真 LLM/Docker。
"""

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from scholar_agent.agent.intent.classifier import IntentClassifier
from scholar_agent.agent.intent.memory import IntentMemoryStore
from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.planner.planner import build_plan
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.agent.workers.coder import CoderAgent
from scholar_agent.agent.workers.data import DataAgent
from scholar_agent.agent.workers.librarian import LibrarianAgent
from scholar_agent.agent.workers.research_coding import ResearchCodingAgent
from scholar_agent.models.event import PlanEvent, PlanEventType
from scholar_agent.models.intent import IntentType
from scholar_agent.models.plan import Plan, PlanStatus
from scholar_agent.models.step import Step, StepStatus
from scholar_agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_STEP_ATTEMPTS = 3
"""单步原地重试上限。防偶发网络/超时抖动重跑一次基本就好；3 次全败
说明是必死任务（repo 404 / 论文太冷门），继续重试只会烧钱——
fail-fast 进 report 收尾。"""

EventSink = Callable[[str, PlanEvent], Awaitable[None]]
"""事件回调签名：(plan_id, event) → None。graph 的 _emit 和 API 的 bus.sink 都是这个。"""


def build_default_agents(llm: LLMClient, sandbox: DockerSandbox) -> dict:
    """组装 4 个业务 Agent（execute_step 按 step.agent 路由用）。"""
    return {
        "librarian_agent": LibrarianAgent(llm),
        "coder_agent": CoderAgent(llm, sandbox),
        "research_coding_agent": ResearchCodingAgent(llm, sandbox),
        "data_agent": DataAgent(llm),
    }


def build_graph(
    llm: LLMClient,
    sandbox: DockerSandbox,
    redis=None,
    event_sink: EventSink | None = None,
    checkpointer=None,
    canceled_plans: set[str] | None = None,
    agents: dict | None = None,
):
    """构建 LangGraph 主流程图（步骤自环版）。

    Args:
        llm: LLM 客户端（或测试 Fake）
        sandbox: Docker 沙箱（或测试 Fake）
        redis: Redis 连接（意图记忆用，None 降级 Noop）
        event_sink: 执行事件回调（SSE 推流用，None 不推）
        checkpointer: LangGraph Checkpointer（生产 Redis 优先 / InMemory 兜底）。
            ⭐ approval 节点的 interrupt() 强依赖它（无处可写挂起点直接
            RuntimeError），走全流程（/runs 入口）的调用方必须传；
            图级直连入口（传入现成 plan）不经过 approval，可不传。
        canceled_plans: 已取消 plan_id 集合。/cancel 端点往里加，
            execute_step 每步开头检查——"当前步骤跑完后生效"语义。
        agents: worker 字典（测试注入 Stub 用；None 走 build_default_agents）
    """
    classifier = IntentClassifier(llm, IntentMemoryStore(redis))
    workers = agents if agents is not None else build_default_agents(llm, sandbox)
    canceled = canceled_plans if canceled_plans is not None else set()

    async def _emit(
        plan_id: str, type_: PlanEventType,
        step: Step | None = None, data: dict | None = None,
    ) -> None:
        """构造事件推给 sink（SSE 用）。sink 挂了不影响主流程。"""
        if event_sink is None or not plan_id:
            return
        event = PlanEvent(type=type_, task_id=step.id if step else None, data=data or {})
        try:
            await event_sink(plan_id, event)
        except Exception:  # noqa: BLE001
            logger.warning("event_sink_failed type=%s", type_)

    # ── 意图 / 规划 / 审批 ──

    async def intent_node(state: AgentState) -> Command:
        """意图识别：三路并行 + 降级。三出口全用 Command 导航。

        - plan 已存在（图级直连入口，测试/内部复用）→ 直进步骤执行。
          语义：计划已在图外确定（含审批），跳过意图/规划/审批。
        - 意图不合格/识别异常 → END（/runs 入口需补发终态事件收 SSE 流）
        - 意图合格 → planner（update 携带识别结果）
        """
        if state.get("plan") is not None:
            return Command(goto="execute_step")  # 图级直连：现成计划

        try:
            intent = await classifier.classify(
                state["user_input"], session_id=state.get("session_id", ""),
            )
        except Exception as exc:  # noqa: BLE001 意图失败 → 终局退出
            logger.error("intent_node_failed: %s", exc)
            # /runs 一段式：SSE 需要终态事件才能收流，这里补发
            await _emit(state.get("plan_id", ""), PlanEventType.plan_failed,
                        None, {"plan_status": "failed", "reason": str(exc)[:200]})
            return Command(goto=END, update={"intent": None,
                                             "error": f"意图识别失败: {exc}"})

        if intent.intent_type == IntentType.unknown:
            await _emit(state.get("plan_id", ""), PlanEventType.plan_failed,
                        None, {"plan_status": "failed", "reason": "unknown_intent"})
            return Command(goto=END, update={"intent": intent})

        return Command(goto="planner", update={"intent": intent})

    async def planner_node(state: AgentState) -> dict:
        """规划：LLM 优先 + 两道闸 + 模板兜底。初始化执行游标。

        revise 回炉时本节点会带着覆写后的 intent 重新规划；
        返回值自带游标重置（current_step=0），新计划从头跑。
        """
        intent = state["intent"]
        plan = await build_plan(llm, intent)
        plan_id = state.get("plan_id")
        if plan_id:
            plan.id = plan_id  # /runs 入口预生成（SSE 已按这个 ID 订阅）
        logger.info("plan_built plan_id=%s steps=%d", plan.id, len(plan.steps))

        return {
            "plan": plan,
            "artifacts": plan.artifacts,
            "current_step": 0,
        }

    async def approval_node(state: AgentState) -> Command:
        """人工审批关卡：interrupt() 挂起整张图，等用户决定。

        ⭐ 节点体只有 interrupt() 一件事（见模块 docstring）：resume 会从
        本节点第一行完整重跑，interrupt 之前不能有任何副作用——不调
        LLM、不发事件、不改 state。plan_awaiting_approval 事件由 API 层
        在 ainvoke 挂起返回后补发（唯一出口，绝不重复）。

        interrupt payload = 审批请求（plan 全量 + 已修订次数），随
        checkpoint 持久化：重启不丢，GET /plans/{id} 可随时查回。

        三出口（resume 值驱动，Command 原子表达去向 + 更新）：
        - approve → execute_step 进入执行循环
        - revise  → 回 planner：feedback 并入 intent.raw_intent（planner
          的 LLM prompt 是完整 intent dump，只写 user_input 重规划看不到）
          重新规划，revision_count 声明增量 1（reducer 累加，/resume 端点据此封顶）
        - abandon（含未知动作兜底）→ report 终态 canceled
        """
        decision = interrupt({
            "plan": state["plan"].model_dump(mode="json"),
            "revision_count": state.get("revision_count") or 0,
        })
        action = decision.get("action", "") if isinstance(decision, dict) else ""

        if action == "approve":
            return Command(goto="execute_step")
        if action == "revise":
            feedback = decision.get("feedback", "")
            intent = state["intent"]
            if feedback:
                # 修改意见并入意图：planner 的 LLM prompt 是完整 intent
                # dump，只写 user_input 重规划根本看不到——追加到原话后面
                # （原话留史、实体/约束保留），新计划才能吸收反馈
                intent.raw_intent = f"{intent.raw_intent}\n[用户修改意见] {feedback}"
            return Command(goto="planner", update={
                "user_input": feedback,
                "intent": intent,  # 随 update 落 checkpoint（内存改不算数）
                "revision_count": 1,  # reducer operator.add 累加
            })
        # abandon：终态标记挂在 plan 上（生命周期唯一权威），report 照此收尾
        plan = state["plan"]
        plan.status = PlanStatus.canceled
        return Command(goto="report", update={"plan": plan})

    # ── 执行循环：execute_step 自环（唯一工位）──

    async def execute_step_node(state: AgentState) -> Command:
        """执行一步：按下标取步骤 → 派工 → 成功自环 / 失败收尾。

        三个出口（Command 原子表达去向 + 更新）：
        1. 取消（/cancel 标记）：剩余步骤标 canceled → report；
        2. 游标越界（全部跑完）→ report；
        3. 跑当前步骤：成功 → 自环（游标 +1 落 checkpoint）；
           失败（3 次全败）→ fail-fast 进 report。

        ⭐ 三条出路都带 "plan"：本节点对 plan 的原地修改（步骤状态、
        产物入库）必须随 update 落 checkpoint 才对 GET /plans/{id}
        可见——checkpointer 只入账返回值里的字段，内存对象上的
        修改不会自动持久化。
        """
        plan = state["plan"]
        idx = state.get("current_step") or 0

        if plan.status == PlanStatus.pending:
            plan.status = PlanStatus.in_progress  # 首次进入执行循环

        if plan.id in canceled:
            # 协作式取消：当前步骤跑到哪算哪，剩余 pending 标 canceled
            _mark_remaining_canceled(plan, idx)
            return Command(goto="report", update={"plan": plan})

        if idx >= len(plan.steps):
            return Command(goto="report", update={"plan": plan})  # 全部跑完

        step = plan.steps[idx]
        ok = await _run_step_with_retry(plan, step)
        if not ok:
            return Command(goto="report", update={"plan": plan})  # fail-fast

        # 自环推进：游标 +1 随 checkpoint 落盘（已完成步骤不重跑）。
        # artifacts 必须随 update 同步——state 字段是 checkpoint 入账的
        # 唯一通道，plan.artifacts 的原地累积不会自动反映到 state["artifacts"]
        return Command(goto="execute_step", update={
            "plan": plan, "artifacts": plan.artifacts,
            "current_step": idx + 1,
        })

    def _mark_remaining_canceled(plan: Plan, from_idx: int) -> None:
        """把 from_idx 之后的 pending 步骤标记为 canceled（剩余部分不执行）。"""
        for step in plan.steps[from_idx:]:
            if step.status == StepStatus.pending:
                step.status = StepStatus.canceled

    async def _run_step_with_retry(plan: Plan, step: Step) -> bool:
        """跑一个步骤（含 3 次原地重试）。返回是否成功。

        未注册 Agent 属确定性错误（重试无意义），直接终局失败；
        执行异常（网络/超时抖动）重试至 MAX_STEP_ATTEMPTS 次。
        """
        agent = workers.get(step.agent)
        if agent is None:
            step.status = StepStatus.failed
            step.error = f"未注册的 Agent: {step.agent}"
            await _emit(plan.id, PlanEventType.task_failed, step,
                        {"error": step.error})
            return False

        step.status = StepStatus.in_progress
        step.started_at = datetime.now(timezone.utc)
        await _emit(plan.id, PlanEventType.task_started, step)

        for attempt in range(1, MAX_STEP_ATTEMPTS + 1):
            try:
                produced = await agent.execute(step, plan.artifacts)
            except Exception as exc:  # noqa: BLE001 Agent 边界统一兜底
                step.error = str(exc)[:500]
                logger.warning("step_failed step=%s attempt=%s error=%s",
                               step.id, attempt, exc)
                if attempt < MAX_STEP_ATTEMPTS:
                    await _emit(plan.id, PlanEventType.task_retrying, step,
                                {"attempt": attempt + 1, "error": step.error})
                continue
            await _finalize_success(plan, step, produced)
            return True

        step.status = StepStatus.failed  # 3 次全败：必死步骤，fail-fast
        await _emit(plan.id, PlanEventType.task_failed, step,
                    {"error": step.error or "unknown"})
        return False

    async def _finalize_success(
        plan: Plan, step: Step, produced: dict,
    ) -> None:
        """成功收尾：产物入账（交接棒递给下游步骤）+ 状态 + 事件。"""
        step.status = StepStatus.completed
        step.error = None
        step.finished_at = datetime.now(timezone.utc)
        step.result = ", ".join(produced.keys())
        plan.artifacts.update(produced)
        await _emit(plan.id, PlanEventType.task_completed, step,
                    {"artifacts": list(produced.keys())})
        for key, art in produced.items():
            await _emit(plan.id, PlanEventType.artifact_created, step,
                        {"artifact_key": key, "type": art.type.value})

    # ── 收尾 ──

    def _decide_terminal_status(plan: Plan) -> PlanStatus:
        """终态判定。优先级：取消 > 全部成功 > 失败。

        canceled 和 completed 一样不算错误（用户主动选择）。
        """
        if plan.id in canceled or plan.status == PlanStatus.canceled:
            return PlanStatus.canceled
        if all(s.status == StepStatus.completed for s in plan.steps):
            return PlanStatus.completed
        return PlanStatus.failed

    def _build_fallback_report(plan: Plan, artifacts: dict) -> str:
        """final_report 缺失（report 步骤失败/被跳过）时的降级摘要。"""
        lines = ["# 执行摘要（降级）", "", f"计划状态：{plan.status.value}"]
        for step in plan.steps:
            lines.append(f"- {step.name}({step.id}): {step.status.value}"
                         + (f" — {step.error}" if step.error else ""))
        for key, art in artifacts.items():
            lines.append(f"\n## {key}\n{str(art.value)[:1000]}")
        return "\n".join(lines)

    async def report_node(state: AgentState) -> dict:
        """收尾：定 plan 终态 + 发终态事件 + 拼最终报告。

        终态判定/降级摘要的逻辑拆在 _decide_terminal_status /
        _build_fallback_report（单一职责，本节点只做编排）。
        """
        plan = state["plan"]
        plan.status = _decide_terminal_status(plan)
        canceled.discard(plan.id)  # 取消标记用完即清（plan 可能重跑）

        # 终态事件三选一：completed / canceled / failed（都进 SSE 终态集合）
        terminal_event = (
            PlanEventType.plan_completed if plan.status == PlanStatus.completed
            else PlanEventType.plan_canceled if plan.status == PlanStatus.canceled
            else PlanEventType.plan_failed
        )
        await _emit(plan.id, terminal_event, None, {"plan_status": plan.status.value})
        error = (
            None if plan.status in (PlanStatus.completed, PlanStatus.canceled)
            else plan.status.value
        )

        artifacts = state.get("artifacts") or plan.artifacts
        # 两条返回路都带 plan：终态必须落 checkpoint（GET /plans/{id}
        # 读的就是这里），否则快照永远停在 in_progress
        if "final_report" in artifacts:
            return {"final_report": str(artifacts["final_report"].value),
                    "error": error, "plan": plan}
        return {"final_report": _build_fallback_report(plan, artifacts),
                "error": error, "plan": plan}

    # ── 组装图 ──
    # 导航分工：intent / approval / execute_step 返回 Command
    # （动态分流，不挂静态出边）；planner / report 走静态边。

    builder = StateGraph(AgentState)
    builder.add_node("intent", intent_node)
    builder.add_node("planner", planner_node)
    builder.add_node("approval", approval_node)
    builder.add_node("execute_step", execute_step_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("intent")
    # intent 的三出口（planner/execute_step/END）由 intent_node 的 Command 决定
    builder.add_edge("planner", "approval")
    # approval 的三出口（execute_step/planner/report）由 approval_node 的
    # Command 决定——resume 值驱动，interrupt 挂起点即恢复点
    # execute_step 的三出口（自环/report）由 execute_step_node 的 Command 决定
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)
