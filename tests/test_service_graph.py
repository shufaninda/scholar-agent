"""graph 编排（步骤自环版，Command 导航 + 游标推进）的功能测试。

覆盖：执行循环（execute_step 自环）的重试 / fail-fast / 取消 /
事件流 / 规则路由路径 / approval interrupt（挂起 + 三动作恢复）/
LangGraph 全流程（Fake LLM + Fake 沙箱）。

执行循环的测试方式：build_graph(agents=Stub) 注入可编程 Stub，
ainvoke 传现成 plan（intent_node 的 Command 直进循环）。

审批流的测试方式：checkpointer=InMemorySaver() 编译，ainvoke 跑到
approval 挂起，ainvoke(Command(resume=决定)) 恢复——thread_id 定位。
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tests.conftest import FakeLLM, make_plan, make_step

from scholar_agent.agent.intent.classifier import IntentClassifier
from scholar_agent.agent.intent.memory import IntentMemoryStore
from scholar_agent.agent.planner.planner import build_plan
from scholar_agent.agent.sandbox.result import SandboxResult
from scholar_agent.graph import build_graph
from scholar_agent.models.artifact import Artifact, ArtifactType
from scholar_agent.models.event import PlanEvent
from scholar_agent.models.plan import PlanStatus
from scholar_agent.models.step import StepStatus


def artifact(key: str, value: str = "v") -> Artifact:
    return Artifact(key=key, value=value, producer_task_id="t")


class StubAgent:
    """可编程 stub Agent：按脚本返回产物或抛异常。"""

    def __init__(self, script: dict[str, list]):
        # script: step_id -> ["ok", "fail", ...] 按次消费
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    async def execute(self, step, artifacts):
        self.calls.append(step.id)
        actions = self.script.get(step.id, ["ok"])
        action = actions.pop(0) if actions else "ok"
        if action == "fail":
            raise RuntimeError(f"{step.id} exploded")
        return {f"out_{step.id}": artifact(f"out_{step.id}", f"{step.id}-result")}


def collect_events() -> tuple[list[PlanEvent], object]:
    events: list[PlanEvent] = []

    async def sink(plan_id: str, event: PlanEvent) -> None:
        events.append(event)

    return events, sink


async def run_loop(
    steps,
    script: dict[str, list] | None = None,
    sink=None,
    canceled: set[str] | None = None,
):
    """跑执行循环图：现成 plan 直进 execute_step 自环。"""
    plan = make_plan(steps)
    stub = StubAgent(script or {})
    graph = build_graph(
        FakeLLM(), None,
        agents={"coder_agent": stub},
        event_sink=sink,
        canceled_plans=canceled,
    )
    result = await graph.ainvoke({"user_input": "test", "plan": plan})
    return plan, result, stub


# ──────────────────────────────────────────────
# 执行循环：execute_step 自环节点
# ──────────────────────────────────────────────


async def test_loop_happy_path():
    """两步串行成功：按数组顺序执行，产物依次入账，事件首尾齐全。"""
    s1 = make_step("s1")
    s2 = make_step("s2")
    events, sink = collect_events()

    plan, result, stub = await run_loop([s1, s2], script={"s1": ["ok"], "s2": ["ok"]}, sink=sink)

    assert plan.status == PlanStatus.completed
    assert s1.status == s2.status == StepStatus.completed
    assert "out_s1" in plan.artifacts and "out_s2" in plan.artifacts
    assert stub.calls == ["s1", "s2"]  # 按数组顺序执行
    assert result["final_report"]  # report 节点拼了报告
    types = [e.type.value for e in events]
    assert types[0] == "task_started" and types[-1] == "plan_completed"


async def test_loop_failure_stops_execution():
    """步骤重试耗尽失败 → fail-fast：下游保持 pending，整图 failed。

    链式下下游全依赖本步，跑下去也必然失败——不再连坐标记
    blocked，剩余步骤保持 pending 原样留给用户看。
    """
    s1 = make_step("s1", outputs=["x"])
    s2 = make_step("s2", required=["x"])
    s3 = make_step("s3")
    events, sink = collect_events()

    plan, result, _ = await run_loop(
        [s1, s2, s3], script={"s1": ["fail", "fail", "fail"]}, sink=sink,
    )

    assert s1.status == StepStatus.failed
    assert s2.status == s3.status == StepStatus.pending  # 下游不执行
    assert "out_s1" not in plan.artifacts  # 失败步骤不写产物
    assert plan.status == PlanStatus.failed
    assert result["error"] == "failed"
    types = [e.type.value for e in events]
    assert "task_retrying" in types and "task_failed" in types and "plan_failed" in types


async def test_loop_retry_then_success():
    """第一次失败第二次成功：产物入账，计划正常完成。"""
    s1 = make_step("s1")
    events, sink = collect_events()

    plan, result, _ = await run_loop([s1], script={"s1": ["fail", "ok"]}, sink=sink)

    assert s1.status == StepStatus.completed
    assert "out_s1" in plan.artifacts
    assert plan.status == PlanStatus.completed
    types = [e.type.value for e in events]
    assert "task_retrying" in types  # 重试事件如实记录


async def test_loop_unknown_agent_fails_immediately():
    """未注册 Agent：确定性错误直接终局失败（重试无意义，1 次都不多跑）。"""
    s1 = make_step("s1", agent="ghost_agent")

    plan, _, stub = await run_loop([s1])

    assert s1.status == StepStatus.failed
    assert stub.calls == []  # Agent 都不存在，execute 从未被调
    assert "未注册" in (s1.error or "")
    assert plan.status == PlanStatus.failed


async def test_loop_cancel_skips_remaining():
    """执行前标记取消：剩余步骤 canceled，Agent 一次都不被调。"""
    s1 = make_step("s1")
    s2 = make_step("s2")
    canceled: set[str] = set()

    plan = make_plan([s1, s2])
    canceled.add(plan.id)  # 执行前就取消 → execute_step 首轮就短路
    stub = StubAgent({})
    graph = build_graph(FakeLLM(), None, agents={"coder_agent": stub},
                        canceled_plans=canceled)

    await graph.ainvoke({"user_input": "test", "plan": plan})

    assert s1.status == s2.status == StepStatus.canceled
    assert plan.status == PlanStatus.canceled
    assert stub.calls == []


async def test_loop_cancel_midway_marks_remaining_only():
    """首步跑完才取消：已完成步骤保持 completed，剩余 canceled。"""
    s1 = make_step("s1")
    s2 = make_step("s2")
    canceled: set[str] = set()

    plan = make_plan([s1, s2])
    stub = StubAgent({})

    # 首步成功后（第 2 次进 execute_step 前）标记取消
    async def cancel_after_first(plan_id: str, event: PlanEvent) -> None:
        if event.type.value == "task_completed":
            canceled.add(plan_id)

    graph = build_graph(FakeLLM(), None, agents={"coder_agent": stub},
                        canceled_plans=canceled, event_sink=cancel_after_first)
    await graph.ainvoke({"user_input": "test", "plan": plan})

    assert s1.status == StepStatus.completed  # 已花的算力不浪费
    assert s2.status == StepStatus.canceled
    assert plan.status == PlanStatus.canceled


# ──────────────────────────────────────────────
# 规则路由路径（intent 直连 classifier，planner 直连 build_plan）
# ──────────────────────────────────────────────


async def test_rule_intent_template_plan():
    """关键词命中的 query 走规则路由 + 模板规划，0 次 LLM 调用。"""
    llm = FakeLLM()
    classifier = IntentClassifier(llm, IntentMemoryStore(None))

    intent = await classifier.classify("复现 Attention Is All You Need 论文")
    plan = await build_plan(llm, intent)

    assert intent.intent_type.value == "Paper_Reproduction"
    assert intent.source == "rule"
    assert [s.type for s in plan.steps] == [
        "paper_parse", "repo_discovery", "code_run", "report",
    ]
    assert llm.calls == []  # 规则命中 + LLM 规划失败回退模板 → 真正 0 调用


# ──────────────────────────────────────────────
# LangGraph 全流程（Fake LLM + Fake 沙箱 + interrupt 审批）
# ──────────────────────────────────────────────


class FakeOneShotSandbox:
    async def execute(self, code: str, timeout: int = 300) -> SandboxResult:
        return SandboxResult(stdout="42", exit_code=0)


def build_full_flow_graph(llm, sink=None):
    """全流程图工厂：带 InMemorySaver（approval interrupt 的底座）。"""
    return build_graph(
        llm, FakeOneShotSandbox(), redis=None, event_sink=sink,
        checkpointer=InMemorySaver(),
    )


async def test_graph_full_flow_with_approval():
    """规则路由 → 模板规划 → approval 挂起 → approve → 执行 → final_report。

    interrupt 语义两阶段验证：
    1. 第一次 ainvoke 在 approval 挂起返回（不是异常），快照 next 含 approval
    2. Command(resume=approve) 唤醒后从 execute_step 继续跑完全程
    """
    llm = FakeLLM(responses=[
        "print(6*7)",       # code_generate（CoderAgent）
        "# 实验报告\n运行成功",  # report（DataAgent）
    ])
    events, sink = collect_events()
    graph = build_full_flow_graph(llm, sink)
    config = {"configurable": {"thread_id": "t-full"}}

    # 阶段 1：跑到 approval 挂起
    await graph.ainvoke({"user_input": "运行一段 Python 代码", "plan_id": "p1"}, config)
    snapshot = await graph.aget_state(config)
    assert "approval" in snapshot.next  # 挂起节点出现在 next 里
    assert snapshot.values["plan"].status == PlanStatus.pending  # 还没执行
    # 挂起前没有任何 task 事件（执行循环未启动）
    assert [e.type.value for e in events] == []

    # 阶段 2：approve 唤醒跑完全程
    result = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    plan = result["plan"]
    assert plan.status.value == "completed"
    assert all(s.status == StepStatus.completed for s in plan.steps)
    assert set(result["artifacts"]) >= {"generated_code", "run_result", "final_report"}
    assert "实验报告" in result["final_report"]
    assert result["artifacts"]["generated_code"].type == ArtifactType.code
    # SSE 事件流覆盖三个步骤的生命周期
    types = [e.type.value for e in events]
    assert types.count("task_completed") == 3
    assert types[-1] == "plan_completed"
    # 终态后不再挂起
    final_snapshot = await graph.aget_state(config)
    assert final_snapshot.next == ()


async def test_graph_approval_abandon_cancels():
    """abandon：不进执行循环，直接 report 终态 canceled。"""
    events, sink = collect_events()
    graph = build_full_flow_graph(FakeLLM(), sink)
    config = {"configurable": {"thread_id": "t-abandon"}}

    await graph.ainvoke({"user_input": "运行一段 Python 代码", "plan_id": "p2"}, config)
    result = await graph.ainvoke(Command(resume={"action": "abandon"}), config)

    assert result["plan"].status == PlanStatus.canceled
    types = [e.type.value for e in events]
    assert types == ["plan_canceled"]  # 只有一个终态事件，零 task 事件


async def test_graph_approval_revise_loops_back_to_planner():
    """revise：回 planner 重新规划 → 再次挂起 → approve 后跑完。

    revision_count 经 reducer 累加（1），新计划游标从零开始。
    """
    llm = FakeLLM(responses=[
        "print(6*7)",       # 第一版 code_generate（被 revise 掉）
        "print(6*7)",       # 第二版 code_generate
        "# 实验报告\n运行成功",  # report
    ])
    graph = build_full_flow_graph(llm)
    config = {"configurable": {"thread_id": "t-revise"}}

    await graph.ainvoke({"user_input": "运行一段 Python 代码", "plan_id": "p3"}, config)

    # revise：feedback 并入 intent（planner 的 LLM prompt 是完整 intent
    # dump，不并入的话重规划根本看不到修改意见），回 planner 重新规划
    result = await graph.ainvoke(
        Command(resume={"action": "revise", "feedback": "改成算 7*6"}), config,
    )
    snapshot = await graph.aget_state(config)
    assert "approval" in snapshot.next  # 重新规划后再次挂起送审
    assert snapshot.values["revision_count"] == 1  # reducer 累加
    assert "改成算 7*6" in snapshot.values["intent"].raw_intent  # 反馈流入意图
    assert result["plan"].status == PlanStatus.pending  # 新计划待执行
    assert snapshot.values["current_step"] == 0  # 游标重置

    # 第二轮 approve 跑完全程
    final = await graph.ainvoke(Command(resume={"action": "approve"}), config)
    assert final["plan"].status == PlanStatus.completed
    assert final["revision_count"] == 1  # 账本保持


async def test_graph_unknown_intent_ends_early():
    """unknown 意图 → 路由 END，不进 planner（没有 plan 产出）。"""
    # "帮我写个贪心算法" 不含任何关键词 → 走 LLM 分类 → FakeLLM 返回空 content
    # → classify 解析失败 → unknown → END（不经过 approval，无需 resume）
    llm = FakeLLM()
    graph = build_graph(llm, FakeOneShotSandbox(), redis=None)

    result = await graph.ainvoke({"user_input": "帮我写个贪心算法"})

    assert result.get("plan") is None
    assert result.get("artifacts") is None
