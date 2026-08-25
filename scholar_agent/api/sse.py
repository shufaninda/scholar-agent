"""SSE 流式推送：EventBus 按 plan_id 分发事件给前端。

对应 Sea 的 SSE 实现。
对应 DESIGN.md 2.2 节"流式执行 + SSE 推送"。

用法：
    bus = EventBus()
    queue = bus.subscribe(plan_id)          # 前端连上 SSE 端点
    await bus.publish(plan_id, event)       # 执行服务推事件
    bus.unsubscribe(plan_id)                # 流结束清理

事件格式（SSE 协议）：
    event: task_started
    data: {"type": "task_started", ...}
"""

import asyncio

from scholar_agent.models.event import PlanEvent, PlanEventType

# 终态事件：收到即结束 SSE 流（canceled 是用户主动选择，同样是终局）
TERMINAL_EVENTS = {
    PlanEventType.plan_completed,
    PlanEventType.plan_failed,
    PlanEventType.plan_canceled,
}

HEARTBEAT_SEC = 15
"""空闲心跳间隔：等待人工审批的时长无界，心跳既防中间代理掐连接，
又让前端确认连接活着（SSE 注释行，EventSource 自动忽略）。"""


def sse_format(event: PlanEvent) -> str:
    """PlanEvent → SSE 文本帧。"""
    return f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"


class EventBus:
    """进程内事件总线：plan_id → asyncio.Queue。

    同一 plan 只支持一个订阅者（简化实现）；没有订阅者时事件直接丢弃
    （SSE 是"尽力推送"，执行主流程不能因为没人看而阻塞）。
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue[PlanEvent]] = {}

    def subscribe(self, plan_id: str) -> asyncio.Queue[PlanEvent]:
        """订阅 plan 的事件流（幂等：重复订阅复用同一队列）。"""
        if plan_id not in self._queues:
            self._queues[plan_id] = asyncio.Queue()
        return self._queues[plan_id]

    def unsubscribe(self, plan_id: str) -> None:
        """取消订阅（流结束/断连时清理）。"""
        self._queues.pop(plan_id, None)

    async def publish(self, plan_id: str, event: PlanEvent) -> None:
        """推事件。无订阅者直接丢（不阻塞执行主流程）。"""
        queue = self._queues.get(plan_id)
        if queue is not None:
            queue.put_nowait(event)

    async def sink(self, plan_id: str, event: PlanEvent) -> None:
        """graph.event_sink 适配器（签名对齐 EventSink，见 graph.py）。"""
        await self.publish(plan_id, event)


async def event_stream(
    bus: EventBus,
    plan_id: str,
    timeout_sec: int = 2400,
):
    """SSE async generator：从队列读事件 yield 文本帧，终态事件后结束。

    ⭐ 心跳语义（审批挂起引入）：等待审批的时长无界，"没事件"不再是
    异常而是常态——空闲 15s 发一条注释行心跳（`: keepalive`），同时
    防中间代理掐连接。旧版"空闲超时发一条假 plan_failed"已删：那会把
    "用户还没点批准"误报成"计划失败"。

    真超时兜底：连续 timeout_sec（默认 40 分钟 ≈ 预算上限）没有任何
    真事件则静默断流——前端发现流断了应走 GET /plans/{id} 查权威
    状态（推拉分工：SSE 尽力推，GET 永远可查）。
    """
    queue = bus.subscribe(plan_id)
    loop = asyncio.get_running_loop()
    last_event_at = loop.time()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SEC)
            except TimeoutError:
                if loop.time() - last_event_at >= timeout_sec:
                    break  # 真异常（执行侧崩溃且无终态事件）：断流，交给 GET 兜底
                yield ": keepalive\n\n"
                continue
            last_event_at = loop.time()
            yield sse_format(event)
            if event.type in TERMINAL_EVENTS:
                break
    finally:
        bus.unsubscribe(plan_id)
