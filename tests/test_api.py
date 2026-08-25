"""API 层功能测试：/runs、/resume、/stream、/cancel。

测试策略：httpx.AsyncClient + ASGITransport 直连 app（不起端口），
dependency_overrides 把 AppState 换成 Fake 组合（FakeLLM + fakeredis +
Stub Agent + InMemorySaver），单测不联网、不花钱、不起容器。

⭐ 审批流测试原理（interrupt 模式）：
    POST /runs 起图 → 轮询 GET /plans/{id} 等 awaiting_approval=True
    （图挂在 approval，快照在 checkpoint 里）→ POST /resume 下决定 →
    轮询等终态。SSE 事件全缓冲在 EventBus 队列里（/runs 时已
    subscribe），终态后 GET /stream 一次读完——不需要实时消费。
"""

import asyncio

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from tests.conftest import FakeLLM

from scholar_agent.api.app_state import AppState
from scholar_agent.api.routes import get_state
from scholar_agent.api.sse import EventBus
from scholar_agent.graph import build_graph
from scholar_agent.main import app
from scholar_agent.models.artifact import Artifact


class StubAgent:
    """永远成功的 stub Agent（模板 plan 的 4 个 assigned_to 都注册它）。

    delay 参数让取消测试能稳定打中断点：慢 Agent 保证 cancel 请求
    落在执行中途（stub 全 instant 的话 cancel 前就跑完了）。
    """

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    async def execute(self, task, artifacts):
        if self.delay:
            await asyncio.sleep(self.delay)
        key = f"out_{task.id}"
        return {key: Artifact(key=key, value=f"{task.id}-result", producer_task_id=task.id)}


def make_state(task_delay: float = 0.0) -> AppState:
    """Fake AppState：规则路由命中意图（不调 LLM）+ 模板规划 + stub 执行。

    checkpointer 必传：approval 的 interrupt() 没有底座直接 RuntimeError
    （生产由 main._build_checkpointer 保证，测试用 InMemorySaver）。
    """
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bus = EventBus()
    canceled: set[str] = set()
    agents = {
        name: StubAgent(task_delay)
        for name in (
            "librarian_agent", "coder_agent",
            "research_coding_agent", "data_agent",
        )
    }
    graph = build_graph(
        FakeLLM(), None, redis=redis,
        event_sink=bus.sink, canceled_plans=canceled, agents=agents,
        checkpointer=InMemorySaver(),
    )
    return AppState(
        graph=graph,
        canceled_plans=canceled,
        bus=bus,
    )


def make_client(state: AppState) -> AsyncClient:
    """挂 dependency override 的异步客户端（用完记得 clear override）。"""
    app.dependency_overrides[get_state] = lambda: state
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client():
    """默认客户端（instant stub）。"""
    state = make_state()
    async with make_client(state) as ac:
        ac.state = state  # 挂到客户端上，测试里直接取
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def state(client) -> AppState:
    """client fixture 顺手暴露 AppState（存问题/模拟竞态用）。"""
    return client.state


# ──────────────────────────────────────────────
# 轮询辅助：interrupt 模式的核心测试手法
# ──────────────────────────────────────────────

async def run_plan(client, user_input="复现 Attention Is All You Need") -> str:
    """POST /runs 起图，返回 plan_id（图将跑到 approval 挂起）。"""
    resp = await client.post("/api/v1/runs", json={"user_input": user_input})
    assert resp.status_code == 202
    return resp.json()["plan_id"]


async def wait_for(client, plan_id, predicate, timeout=10) -> dict:
    """轮询 GET /plans/{id}（checkpoint 权威）直到谓词满足。"""
    for _ in range(timeout * 20):
        plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
        if predicate(plan):
            return plan
        await asyncio.sleep(0.05)
    raise TimeoutError(f"plan {plan_id} 未满足条件，当前: {plan}")


def is_awaiting(plan) -> bool:
    return bool(plan.get("awaiting_approval"))


def is_terminal(plan) -> bool:
    return plan.get("status") in ("completed", "failed", "canceled")


async def wait_awaiting(client, plan_id, expect_rc: int | None = None) -> dict:
    """等待图挂起在 approval 且后台 resume 任务已完全收尾。

    两个谓词条件各堵一个竞态窗口：
    expect_rc：resume(revise) 派发后、图重新挂起前，快照仍是旧挂起
    （revision_count 未入账）——旧挂起会骗过轮询。带 expect_rc 时
    额外要求 revision_count 达到该值，确保等到 revise 回炉后的
    **新一轮**挂起。
    resuming 收尾：checkpoint 写入新挂起点的时刻**早于**后台任务
    ainvoke 返回（finally 释放 resuming_runs）——谓词若只看快照，
    会在释放前穿透，紧跟的下一个 resume 被双击守卫误伤 409。
    额外要求 resuming_runs 已清空，即上一轮 resume 完全落地。
    """
    state = client.state

    def pred(plan) -> bool:
        if not is_awaiting(plan):
            return False
        if plan_id in state.resuming_runs:
            return False  # 后台任务还在收尾，快照可见≠可安全 resume
        return expect_rc is None or plan.get("revision_count") == expect_rc

    return await wait_for(client, plan_id, pred)


async def wait_terminal(client, plan_id) -> dict:
    return await wait_for(client, plan_id, is_terminal)


async def resume(client, plan_id, body) -> int:
    """POST /resume 下审批决定，返回状态码。"""
    resp = await client.post(f"/api/v1/plans/{plan_id}/resume", json=body)
    return resp.status_code


async def run_and_approve(client) -> str:
    """/runs + approve，一路跑到终态（多数测试的前置动作）。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)
    assert await resume(client, plan_id, {"action": "approve"}) == 202
    await wait_terminal(client, plan_id)
    return plan_id


# ──────────────────────────────────────────────
# 基础链路
# ──────────────────────────────────────────────


async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "0.1.0"}


async def test_ping_router_mounted(client):
    resp = await client.get("/api/v1/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


async def test_runs_rejects_empty_input(client):
    resp = await client.post("/api/v1/runs", json={"user_input": ""})
    assert resp.status_code == 422  # pydantic min_length 校验


# ──────────────────────────────────────────────
# POST /runs：起图 + approval 挂起
# ──────────────────────────────────────────────


async def test_run_suspends_at_approval(client):
    """一段式起图：意图 → 规划 → approval interrupt 挂起。

    挂起语义：awaiting_approval=True、status 仍 pending（未进执行
    循环）、revision_count=0、plan 内容可查（checkpoint 快照）。
    """
    plan_id = await run_plan(client)
    plan = await wait_awaiting(client, plan_id)

    assert plan["status"] == "pending"  # 还没进执行循环
    assert plan["awaiting_approval"] is True
    assert plan["revision_count"] == 0
    assert len(plan["steps"]) >= 1  # 模板规划出了步骤
    # 图真的停着：不 resume 永远不到终态（等 0.3s 验证）
    await asyncio.sleep(0.3)
    refreshed = (await client.get(f"/api/v1/plans/{plan_id}")).json()
    assert refreshed["status"] == "pending"


async def test_get_plan_404(client):
    resp = await client.get("/api/v1/plans/no-such-plan")
    assert resp.status_code == 404


# ──────────────────────────────────────────────
# POST /resume：三动作
# ──────────────────────────────────────────────


async def test_resume_approve_completes(client):
    """approve：唤醒图跑执行循环 → 终态 completed，meta 统计正确。"""
    plan_id = await run_and_approve(client)

    plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
    assert plan["status"] == "completed"
    assert plan["awaiting_approval"] is False  # 终态不再是挂起
    statuses = [s["status"] for s in plan["steps"]]  # 进度直接由步骤状态推导
    assert statuses and set(statuses) == {"completed"}


async def test_resume_revise_replans_and_suspends_again(client):
    """revise：回 planner 重新规划 → 再次挂起，revision_count 累加。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)

    ok = await resume(client, plan_id, {"action": "revise", "feedback": "加上消融实验"})
    assert ok == 202

    # 回炉重新规划后又挂起在 approval（新一轮送审）；
    # expect_rc=1 确保等到的是 revise 后的新挂起而非旧挂起
    plan = await wait_awaiting(client, plan_id, expect_rc=1)
    assert plan["revision_count"] == 1  # reducer 累加
    assert plan["status"] == "pending"  # 新计划仍是待执行

    # 第二轮 approve 收尾
    assert await resume(client, plan_id, {"action": "approve"}) == 202
    final = await wait_terminal(client, plan_id)
    assert final["status"] == "completed"


async def test_resume_abandon_cancels(client):
    """abandon：终态 canceled（用户主动放弃，不算失败）。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)

    assert await resume(client, plan_id, {"action": "abandon"}) == 202

    plan = await wait_terminal(client, plan_id)
    assert plan["status"] == "canceled"


async def test_resume_unknown_plan_404(client):
    resp = await client.post(
        "/api/v1/plans/none/resume", json={"action": "approve"}
    )
    assert resp.status_code == 404


async def test_resume_not_awaiting_409(client):
    """终态后再 resume（重复 approve）→ 409：状态机只允许挂起态恢复。"""
    plan_id = await run_and_approve(client)
    assert await resume(client, plan_id, {"action": "approve"}) == 409


async def test_resume_revise_requires_feedback(client):
    """revise 空 feedback → 422（重新规划没有输入等于原地打转）。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)
    assert await resume(client, plan_id, {"action": "revise"}) == 422


async def test_resume_double_click_409(client, state):
    """双击守卫：resume 处理中（resuming_runs 占位）第二次请求 → 409。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)

    state.resuming_runs.add(plan_id)  # 模拟第一次 resume 还在处理
    assert await resume(client, plan_id, {"action": "approve"}) == 409
    state.resuming_runs.discard(plan_id)  # 清理，不影响后续


async def test_resume_revision_limit(client):
    """修订上限：3 次 revise 用尽后第 4 次 → 409（防无限重规划烧账单）。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)

    for i in range(1, 4):  # 第 1/2/3 次都放行（等新一轮挂起：rc 逐次 +1）
        assert await resume(client, plan_id, {"action": "revise", "feedback": "再改"}) == 202
        await wait_awaiting(client, plan_id, expect_rc=i)
    # 第 4 次：revision_count=3 已达 MAX_REVISIONS
    assert await resume(client, plan_id, {"action": "revise", "feedback": "还改"}) == 409
    # abandon 仍然可用（上限只拦 revise，不拦退出）
    assert await resume(client, plan_id, {"action": "abandon"}) == 202


# ──────────────────────────────────────────────
# GET /stream：SSE 全流程
# ──────────────────────────────────────────────


async def test_stream_covers_approval_and_execution(client):
    """SSE 全链路：审批事件 → 任务事件 → 终态事件（一次读完缓冲）。"""
    plan_id = await run_and_approve(client)

    resp = await client.get(f"/api/v1/plans/{plan_id}/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    event_types = [
        line.removeprefix("event: ")
        for line in resp.text.splitlines()
        if line.startswith("event: ")
    ]
    # 审批事件在任务事件之前（挂起时 API 层补发）
    assert "plan_awaiting_approval" in event_types
    assert event_types.index("plan_awaiting_approval") < event_types.index("task_started")
    assert "task_completed" in event_types
    assert "artifact_created" in event_types
    assert event_types[-1] == "plan_completed"  # stub 全成功


async def test_stream_unknown_plan_404(client):
    resp = await client.get("/api/v1/plans/none/stream")
    assert resp.status_code == 404


# ──────────────────────────────────────────────
# POST /cancel：三分支
# ──────────────────────────────────────────────


async def test_cancel_unknown_plan_404(client):
    resp = await client.post("/api/v1/plans/none/cancel")
    assert resp.status_code == 404


async def test_cancel_while_awaiting_translates_to_abandon(client):
    """挂起中取消：转发 abandon → 立即终态 canceled（无需用户先 approve）。"""
    plan_id = await run_plan(client)
    await wait_awaiting(client, plan_id)

    resp = await client.post(f"/api/v1/plans/{plan_id}/cancel")
    assert resp.status_code == 200
    assert resp.json() == {"plan_id": plan_id, "canceled": True, "mode": "abandon"}

    plan = await wait_terminal(client, plan_id)
    assert plan["status"] == "canceled"


async def test_cancel_while_running_is_graceful():
    """执行中取消：慢 Agent（0.5s/任务）保证 cancel 打断在执行中，
    软取消后剩余任务标 canceled，plan 终态 canceled（而非 completed）。"""
    state = make_state(task_delay=0.5)
    async with make_client(state) as client:
        client.state = state  # wait_awaiting 谓词要读 resuming_runs
        plan_id = await run_plan(client)
        await wait_awaiting(client, plan_id)
        await resume(client, plan_id, {"action": "approve"})
        # 模板 4 个任务 × 0.5s：第 1 个刚开始就取消，后 3 个必被跳过
        resp = await client.post(f"/api/v1/plans/{plan_id}/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"plan_id": plan_id, "canceled": True, "mode": "graceful"}

        plan = await wait_terminal(client, plan_id)
        assert plan["status"] == "canceled"
        statuses = {s["status"] for s in plan["steps"]}
        assert "canceled" in statuses  # 有步骤被跳过
    app.dependency_overrides.clear()


async def test_cancel_terminal_409(client):
    """终态后取消 → 409：没有可取消的东西。"""
    plan_id = await run_and_approve(client)
    resp = await client.post(f"/api/v1/plans/{plan_id}/cancel")
    assert resp.status_code == 409
