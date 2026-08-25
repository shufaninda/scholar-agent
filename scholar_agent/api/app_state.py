"""应用状态组装：把 graph / services / bus / store 挂到一个对象上，供路由依赖注入。

对应 Sea 的依赖注入容器（wire 生成的 ProviderSet）。

为什么集中组装：
    1. lifespan 里建一次，所有路由共享（单例）
    2. 测试不用起真 LLM/Docker——override 依赖函数返回 Fake AppState 即可
    3. 资源生命周期清晰：启动建、关闭拆

⭐ 审批重构后的状态管理（只剩一套）：plan 的生命周期状态全部住在
LangGraph checkpoint 里（thread_id = plan_id），本项目不再持有
PlanRegistry——Sea 时代"内存 dict 存计划 + 两段式 REST"的组合已删。
GET /plans/{id}、/resume、/cancel 都从 aget_state 读权威状态。
"""

import asyncio

from scholar_agent.api.sse import EventBus
from scholar_agent.graph import build_graph


class AppState:
    """路由层共享资源的容器（app.state.scholar）。

    Attributes:
        graph: LangGraph 主图（Agent 循环版，event_sink 已接 EventBus，
            编译期已挂 checkpointer——审批 interrupt 的挂起点存这里。
            intent/planner 的业务逻辑在图内经 PlanService 门面调用，
            API 层不需要单独持有）
        canceled_plans: 已取消 plan_id 集合（/cancel 写，execute_step 每步读）
        active_runs: 初始 run（POST /runs 起的全流程图）在跑的 plan_id 集合。
            含 SSE 竞态占位（返回 202 前先加，防前端连流 404）
        resuming_runs: resume/abandon 后台任务在跑的 plan_id 集合。
            双击 approve 的 409 守卫靠它——与 active_runs 分开是因为
            两者窗口重叠但语义不同：图挂起后初始任务还在收尾
            （发事件/清标记），此时 resume 完全合法（图状态已在
            checkpoint，收尾不会再碰图），不能误伤
        bus: SSE 事件总线
        tasks: 后台执行任务强引用集合（防 GC 回收运行中的协程）
    """

    def __init__(
        self,
        graph,
        canceled_plans: set[str],
        bus: EventBus,
    ):
        self.graph = graph
        self.canceled_plans = canceled_plans
        self.active_runs: set[str] = set()
        self.resuming_runs: set[str] = set()
        self.bus = bus
        self.tasks: set[asyncio.Task] = set()

    def spawn(self, coro) -> asyncio.Task:
        """起后台任务并保留强引用（asyncio 只持弱引用，裸 create_task 会被 GC）。"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task


def build_app_state(llm, sandbox, redis, checkpointer=None) -> AppState:
    """从底层依赖组装 AppState（main.py lifespan 用；测试传 Fake）。

    checkpointer：审批 interrupt 的持久化底座，None 时图仍可编译但
    /runs 全流程会在 approval 节点 RuntimeError——生产环境必须传
    （main._build_checkpointer 保证任何降级场景都有口袋）。
    """
    bus = EventBus()
    # 取消标记集合：/cancel 端点写入，graph 的 execute_step 每步检查
    canceled: set[str] = set()
    graph = build_graph(
        llm, sandbox, redis=redis, event_sink=bus.sink, canceled_plans=canceled,
        checkpointer=checkpointer,
    )
    return AppState(
        graph=graph,
        canceled_plans=canceled,
        bus=bus,
    )
