"""API 路由：/runs、/resume、/stream、/cancel。

完整端点一览：
- POST /runs                      一段式：起全流程图（意图→规划→审批→执行→报告）
- POST /plans/{plan_id}/resume    审批决定（approve/revise/abandon），唤醒挂起的图
- GET  /plans/{plan_id}           查询 plan 状态（checkpoint 权威，轮询兜底）
- GET  /plans/{plan_id}/stream    SSE 事件流（前端 EventSource 消费）
- POST /plans/{plan_id}/cancel    取消：执行中软取消 / 审批挂起转 abandon
- GET  /ping                      链路探活

⭐ 状态唯一权威 = LangGraph checkpoint（thread_id = plan_id）：
    旧两段式（POST /plans 存 registry + POST /execute 发起执行）已删。
    plan 的生命周期状态全部住在 checkpoint 里——GET /resume /cancel
    都从 aget_state 读权威状态，重启不丢、天然断点续跑。
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from scholar_agent.api.app_state import AppState
from scholar_agent.api.sse import event_stream
from scholar_agent.core.logging import logger
from scholar_agent.models.event import PlanEvent, PlanEventType

router = APIRouter(tags=["plans"])

MAX_REVISIONS = 3
"""revise（带反馈重新规划）的次数上限。改了 3 版还不满意说明需求本身
模糊，引导用户 abandon 重开。上限拦在 API 层（调用侧守门），
图内 revision_count 只如实记账（reducer 累加）——职责分工。"""


# ──────────────────────────────────────────
# 请求 / 响应模型
# ──────────────────────────────────────────

class CreatePlanRequest(BaseModel):
    """POST /runs 请求体。"""

    user_input: str = Field(min_length=1, description="用户原始输入")
    session_id: str = Field(default="", description="会话 ID（意图记忆用）")


class ExecuteAccepted(BaseModel):
    """POST /runs 的 202 响应。"""

    plan_id: str
    status: str = "in_progress"


class ResumeRequest(BaseModel):
    """POST /plans/{id}/resume 请求体。"""

    action: Literal["approve", "revise", "abandon"] = Field(description="审批动作")
    feedback: str = Field(default="", description="revise 时的修改意见（并入 intent 送重规划）")


# ──────────────────────────────────────────
# 依赖注入：app.state.scholar → 路由参数
# ──────────────────────────────────────────

def get_state(request: Request) -> AppState:
    """从 app.state 取 AppState。测试用 dependency_overrides 换成 Fake。"""
    return request.app.state.scholar


# ──────────────────────────────────────────
# checkpoint 快照工具（routes 内部共享）
# ──────────────────────────────────────────

def _thread_config(plan_id: str) -> dict:
    """checkpointer 寻址：thread_id = plan_id（断点续跑的钥匙）。"""
    return {"configurable": {"thread_id": plan_id}}


async def _get_snapshot(state: AppState, plan_id: str):
    """读 checkpoint 快照（thread 不存在时 values 为空）。"""
    return await state.graph.aget_state(_thread_config(plan_id))


def _plan_from_snapshot(snapshot):
    """从快照提取 plan；不存在（未跑过/纯随机 ID）返回 None。"""
    if snapshot is None:
        return None
    return (snapshot.values or {}).get("plan")


def _is_awaiting(snapshot) -> bool:
    """图是否挂起在 approval 等 resume。

    挂起节点的名字会出现在 snapshot.next 里——这是"挂起中"
    的权威判定，GET/ resume / cancel 三处复用。
    """
    return snapshot is not None and "approval" in (snapshot.next or ())


async def _emit_awaiting_if_suspended(state: AppState, plan_id: str) -> None:
    """图挂起在 approval 时补发 plan_awaiting_approval 事件（幂等出口）。

    为什么在 API 层发而不是 approval 节点里发：resume 会从挂起
    节点第一行完整重跑，节点里发事件 = 每次恢复重复推一次；API 层
    在 ainvoke 返回后检查快照补发，前后台任务共用这一个出口。
    """
    snapshot = await _get_snapshot(state, plan_id)
    if not _is_awaiting(snapshot):
        return
    plan = _plan_from_snapshot(snapshot)
    if plan is None:
        return
    await state.bus.publish(plan_id, PlanEvent(
        type=PlanEventType.plan_awaiting_approval,
        data={
            "plan": plan.model_dump(mode="json"),
            "revision_count": (snapshot.values or {}).get("revision_count") or 0,
        },
    ))


async def _fail_sse(state: AppState, plan_id: str, reason: str) -> None:
    """后台任务异常兜底：补发 plan_failed 让 SSE 收流（否则前端挂到超时）。"""
    await state.bus.publish(plan_id, PlanEvent(
        type=PlanEventType.plan_failed,
        data={"plan_status": "failed", "reason": reason[:200]},
    ))


# ──────────────────────────────────────────
# 后台任务体
# ──────────────────────────────────────────

async def _run_full_flow(
    state: AppState, plan_id: str, user_input: str, session_id: str,
) -> None:
    """后台任务体：跑全流程图。interrupt 挂起时 ainvoke 正常返回
    （不是异常），返回后检查快照补发审批事件再退出。
    """
    try:
        await state.graph.ainvoke(
            {"user_input": user_input, "session_id": session_id, "plan_id": plan_id},
            config=_thread_config(plan_id),
        )
    except Exception as exc:  # noqa: BLE001 后台任务异常必须兜底
        logger.error("background_run_failed plan=%s error=%s", plan_id, exc)
        await _fail_sse(state, plan_id, str(exc))
        return
    finally:
        # 占位释放必须在 ainvoke 返回后同步完成（中间不能有 await）：
        # 图状态此刻已落 checkpoint，轮询方看到挂起即可发下一个请求，
        # 若释放滞后于事件补发（含 await 让出点）会被守卫误伤 409
        state.active_runs.discard(plan_id)
    await _emit_awaiting_if_suspended(state, plan_id)


async def _resume_run(state: AppState, plan_id: str, decision: dict) -> None:
    """后台任务体：Command(resume=决定) 唤醒挂起的图跑到终态。

    revise 分支回 planner 重新规划后会再次挂起在 approval——
    结束时同样补发审批事件（前端看到新版计划继续审）。
    占位用 resuming_runs（不是 active_runs）：双击守卫只看这里，
    初始 run 的收尾窗口不会误伤 resume（见 app_state.py）。
    """
    state.resuming_runs.add(plan_id)
    try:
        await state.graph.ainvoke(
            Command(resume=decision), config=_thread_config(plan_id),
        )
    except Exception as exc:  # noqa: BLE001 后台任务异常必须兜底
        logger.error("background_resume_failed plan=%s error=%s", plan_id, exc)
        await _fail_sse(state, plan_id, str(exc))
        return
    finally:
        # 同 _run_full_flow：ainvoke 返回即释放（零 await 窗口），
        # 事件补发放在释放之后。注意 checkpoint 写入新挂起点的时刻仍
        # 早于本释放（框架内部收尾有 await 间隙）——轮询方在这个极小
        # 窗口内 resume 会吃 409（双击守卫误伤），客户端重试即可
        state.resuming_runs.discard(plan_id)
    await _emit_awaiting_if_suspended(state, plan_id)


# ──────────────────────────────────────────
# 端点
# ──────────────────────────────────────────

@router.get("/ping")
async def ping() -> dict:
    """链路探活：验证 router 挂载成功（/health 在 main.py，不重复）。"""
    return {"pong": True}


@router.post("/runs", status_code=202)
async def run_one_shot(
    body: CreatePlanRequest,
    state: AppState = Depends(get_state),
):
    """一段式入口：意图 → 规划 → approval 挂起（interrupt）→ 等 resume。

    plan_id 由 API 层预生成（SSE 队列必须先于图执行订阅，而 plan
    要等 planner_node 才诞生——ID 反过来传进图里）。返回 202 后前端
    立即连 SSE：active_runs 先占位，stream 端点的存在性检查才不会
    在"图还没起跑"的窗口误报 404。
    """
    plan_id = str(uuid.uuid4())
    state.bus.subscribe(plan_id)   # 先订阅（幂等，队列缓冲）
    state.active_runs.add(plan_id)  # 占位（防 SSE 竞态 404）
    state.spawn(_run_full_flow(state, plan_id, body.user_input, body.session_id))
    return ExecuteAccepted(plan_id=plan_id)


@router.post("/plans/{plan_id}/resume", status_code=202)
async def resume_plan(
    plan_id: str,
    body: ResumeRequest,
    state: AppState = Depends(get_state),
):
    """审批决定入口：Command(resume=决定) 唤醒挂起的图。

    三动作（即 approval 节点 interrupt() 的返回值）：
    - approve：进执行循环
    - revise：回 planner 重新规划（feedback 并入 intent 送规划）
    - abandon：终态 canceled
    """
    snapshot = await _get_snapshot(state, plan_id)
    if _plan_from_snapshot(snapshot) is None:
        raise HTTPException(status_code=404, detail="plan 不存在")
    if not _is_awaiting(snapshot):
        raise HTTPException(status_code=409, detail="计划不在等待审批状态")
    if plan_id in state.resuming_runs:
        raise HTTPException(status_code=409, detail="已有 resume 在处理（双击守卫）")
    _validate_revision(snapshot, body)

    decision = {"action": body.action, "feedback": body.feedback}
    state.spawn(_resume_run(state, plan_id, decision))
    return {"plan_id": plan_id, "status": "resuming", "action": body.action}


def _validate_revision(snapshot, body: ResumeRequest) -> None:
    """revise 前置校验：反馈非空 + 修订次数没超上限（提前返回风格）。"""
    if body.action != "revise":
        return
    if not body.feedback.strip():
        raise HTTPException(status_code=422, detail="revise 需要非空 feedback")
    used = (snapshot.values or {}).get("revision_count") or 0
    if used >= MAX_REVISIONS:
        raise HTTPException(
            status_code=409,
            detail=f"修订次数已用尽（{used}/{MAX_REVISIONS}），请 abandon 后重新发起",
        )


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str, state: AppState = Depends(get_state)):
    """查询 plan 状态：checkpoint 权威快照（SSE 之外的第二通道）。

    awaiting_approval=True 时前端应渲染审批 UI——挂起中图不动，
    快照永远可查（重启也不丢，checkpoint 统一状态的收益）。
    """
    snapshot = await _get_snapshot(state, plan_id)
    plan = _plan_from_snapshot(snapshot)
    if plan is None:
        raise HTTPException(status_code=404, detail="plan 不存在")
    result = plan.model_dump(mode="json")
    result["awaiting_approval"] = _is_awaiting(snapshot)
    result["revision_count"] = (snapshot.values or {}).get("revision_count") or 0
    return result


@router.get("/plans/{plan_id}/stream")
async def plan_stream(plan_id: str, state: AppState = Depends(get_state)):
    """SSE 事件流：task_* / artifact_created / plan_* 事件，终态后自动断开。

    审批挂起期间可能长时间无事件——15s 心跳保活（见 sse.py）。
    """
    if not await _plan_exists(state, plan_id):
        raise HTTPException(status_code=404, detail="plan 不存在")
    return StreamingResponse(
        event_stream(state.bus, plan_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/plans/{plan_id}/cancel")
async def cancel_plan(plan_id: str, state: AppState = Depends(get_state)):
    """取消：按图所处阶段分流（语义不同，殊途同归 canceled 终态）。

    - 审批挂起且无人恢复：转发 abandon（resume 唤醒 → report 终态
      canceled）。挂起判定优先于执行中判定——图挂起后初始任务可能
      还在收尾（active_runs 有占位），但图状态已在 checkpoint，
      转发 abandon 完全安全
    - 执行中（初始 run 或 resume 后）：软取消——标记集合，
      execute_step 下一轮检查生效（正在跑的步骤不中断，已花的算力
      不浪费）；若 resume 在途（approve 未生效），图被唤醒后同样
      在 execute_step 处停下
    - 已终态：409（没有可取消的东西）
    """
    snapshot = await _get_snapshot(state, plan_id)
    in_flight = plan_id in state.active_runs or plan_id in state.resuming_runs
    if _plan_from_snapshot(snapshot) is None and not in_flight:
        raise HTTPException(status_code=404, detail="plan 不存在")
    if _is_awaiting(snapshot) and plan_id not in state.resuming_runs:
        state.spawn(_resume_run(state, plan_id, {"action": "abandon", "feedback": ""}))
        return {"plan_id": plan_id, "canceled": True, "mode": "abandon"}
    if in_flight:
        state.canceled_plans.add(plan_id)  # 软取消：下一任务开始前生效
        return {"plan_id": plan_id, "canceled": True, "mode": "graceful"}
    raise HTTPException(status_code=409, detail="计划已终态，无需取消")


async def _plan_exists(state: AppState, plan_id: str) -> bool:
    """存在性：在跑集合（初始/resume）或 checkpoint 快照（挂起/终态）。"""
    if plan_id in state.active_runs or plan_id in state.resuming_runs:
        return True
    return _plan_from_snapshot(await _get_snapshot(state, plan_id)) is not None
