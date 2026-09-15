# Scholar-Agent：论文复现多 Agent 系统（LangGraph 版）

一句话定位：**用 LangGraph 编排的多 Agent 论文复现系统——LLM 负责"建议"，代码负责"约束"，关键决策交还给人。**

对标 Go 语言的 Sea-mult-agent 项目，用 Python + LangGraph 重写。不是翻译，是重新设计：Go 版自写的 DAG 调度器、叫号器、三本账，在 LangGraph 里全部退化成一个自环节点 + 一个游标。

📖 **深入阅读**：[`新手全景导读.md`](新手全景导读.md) —— 单文件百科：架构全景 + 八站全链路源码走读 + 三大亮点 + 面试速答（10 问）。

---

## 0. 三大亮点（先看这里）

| 亮点 | 一句话 | 代码位置 |
|---|---|---|
| **① 代码约束 LLM** | LLM 只能"提议"，出什么步骤/派给谁/要不要重试/跑没跑成，全部由确定性代码裁决 | `planner/validator.py` + `graph.py` |
| **② 多 Agent + 意图识别** | 4 个异构 Agent、三路并行意图识别（分类/重写/抽取）、artifacts 全局仓库做消息总线 | `intent/classifier.py` + `workers/` |
| **③ Sandbox + Harness 自愈** | Docker 两阶段网络沙箱（装依赖联网、跑代码断网）+ 5 道闸自愈引擎（防 LLM 越权/偷改/造假） | `sandbox/` + `harness/runner.py` |

这三个亮点互为犄角：意图识别决定"做什么"，代码约束决定"听谁的"，Harness 沙箱决定"在哪做、做错了怎么办"。剩下的人工审批（HITL）用 LangGraph 原生 `interrupt()` 实现，状态管理只剩 checkpoint 一套账本。

---

## 1. 架构总览

### 1.1 主图（5 个节点，编译期固定）

```
intent ──Command──► planner ──► approval ──approve──► execute_step ──┐
  │ （plan 已存在：图级直连入口）  ▲（interrupt 挂起）   ▲            │（还有步骤）
  │  └──Command──► execute_step   │（revise：带新输入   │            │（自环：游标+1）
  │                               │  回炉重新规划）      │            │
  └─（意图不合格）─Command──► END  └─ abandon ─► report ◄────────────┘
                                                    ▲（走完/失败/取消）
                                                report ──► END
```

| 节点 | 职责 | LLM 参与度 |
|---|---|---|
| `intent` | 意图识别（规则快速路径 + 三路并行 LLM） | 高（三路调用） |
| `planner` | 规划（LLM 优先 → 两道闸 → 模板兜底） | 中（提议，校验在代码） |
| `approval` | 人工审批（`interrupt()` 挂起整图） | 零（等人） |
| `execute_step` | 执行一步（自环推进游标） | 零（Agent 墙内才可能有 LLM） |
| `report` | 收尾（定终态 + 拼报告） | 零 |

**导航分工**：返回 `Command` 的动态分流节点（intent / approval / execute_step）不挂静态出边；固定路径（planner→approval、report→END）用静态边。

### 1.2 为什么不用 DAG（步骤数组替代的核心论证）

论文复现的任务依赖恒为"前一步"：解析论文 → 搜 repo → 跑代码 → 报告。**业务图 100% 是链式。链是 DAG 的特例：数组下标同时承载拓扑序、依赖序、执行序（四序合一）**，独立的 `dependencies` / `TaskEdge` / 拓扑排序 / 分层并行全部冗余。

Go 版需要 DAG 是因为它的调度器是通用的（要支持任意形状的任务图）；本项目业务恒为链式，砍掉通用性换来的是：

| 原 DAG 版（Go 移植版） | 现步骤版 |
|---|---|
| 7 个图节点（pick_task/worker/verify 循环） | 5 个图节点（execute_step 自环） |
| TaskNode + TaskEdge + PlanGraph 三套模型 | Step + Plan 两套 |
| 8 种任务状态（含 ready/blocked/skipped） | 5 种步骤状态 |
| 拓扑排序 + 叫号器 + 三本账（约 300 行） | 游标 + 计数器（约 60 行） |
| 连坐机制（失败标 blocked 下游） | fail-fast（下游保持 pending） |

### 1.3 目录结构（⭐ = 核心文件）

```
scholar_agent/
├── graph.py                  # ⭐ 主图：5 节点 + execute_step 自环
├── state.py                  # AgentState（含 current_step 游标）
├── models/
│   ├── step.py               # ⭐ Step 模型（工单）
│   ├── plan.py               # Plan 模型（工单本 + 仓库）
│   ├── artifact.py           # Artifact 模型（交接棒）
│   ├── intent.py             # IntentContext / IntentType
│   └── harness.py            # HarnessAttempt / HarnessReport
├── agent/
│   ├── intent/
│   │   ├── classifier.py     # ⭐ 三路并行意图识别
│   │   ├── rule_router.py    # 规则快速路径（关键词命中不调 LLM）
│   │   └── memory.py         # Redis 意图记忆（降级 Noop）
│   ├── planner/
│   │   ├── planner.py        # 规划主入口（LLM 优先 + 模板兜底）
│   │   ├── agent_planner.py  # LLM 规划（Structured Outputs）
│   │   ├── validator.py      # ⭐ 两道闸（约束 LLM 的核心）
│   │   └── templates.py      # 确定性模板（永远可用的兜底）
│   ├── workers/
│   │   ├── base.py           # BaseAgent
│   │   ├── librarian.py      # 论文解析（纯 LLM）
│   │   ├── coder.py          # 代码生成/执行（LLM + 沙箱）
│   │   ├── research_coding.py# ⭐ repo 发现/执行（确定性 + Harness）
│   │   └── data.py           # 报告生成（纯 LLM）
│   ├── harness/
│   │   ├── runner.py         # ⭐ 自愈引擎主循环（5 道闸）
│   │   ├── patch_policy.py   # 闸1：补丁静态校验
│   │   ├── fingerprint.py    # 闸3：SHA256 + 仓库指纹
│   │   ├── metrics_recompute.py # 闸4：指标重算防伪
│   │   ├── circuit_breaker.py   # 闸5：熔断器
│   │   └── feedback_injector.py # 负反馈注入
│   ├── sandbox/
│   │   ├── docker_executor.py# ⭐ 两阶段网络沙箱
│   │   ├── registry.py       # 容器索引（Redis）
│   │   └── security.py       # 资源限制
│   ├── prompts/              # 各处 system/user prompt
│   └── llm_client.py         # LLM 客户端封装
├── api/
│   ├── routes.py             # ⭐ 6 个端点
│   ├── sse.py                # SSE 事件总线（15s 心跳）
│   └── app_state.py          # 依赖注入容器
└── main.py                   # FastAPI 组装 + lifespan
```

---

## 2. 数据模型：三个核心结构的血缘

### 2.1 Step = 一张工单（`models/step.py`）

```python
class Step(BaseModel):
    id: str                    # "s1"，SSE 事件归位用
    name: str                  # "解析论文"，前端展示
    type: str                  # paper_parse / repo_discovery / code_run / report
    agent: str                 # ⭐ 派工钥匙：workers 名册查表路由
    status: StepStatus         # 运行时状态（只由代码填）
    required_artifacts: list[str]  # 领什么料（必须有 producer）
    output_artifacts: list[str]    # 交什么货（跨步骤不能重名）
    inputs: dict               # 执行参数（planner 确定性填充）
    result / error / started_at / finished_at  # 运行时痕迹
```

**白话**：planner 是包工头，Step 是他开的工单。`agent` 决定"哪个工人来干"（查表路由，不是 LLM 说了算），`required/output_artifacts` 决定"领什么料、交什么货"（validator 校验闭合）。

**LLM 边界**（"代码约束 LLM"在数据模型层的落地）：
- LLM 只能填"设计字段"：name / type / agent / artifacts 契约，且必须过 validator 两道闸；
- status / result / error 是运行时状态，全部由代码（execute_step 节点）填写，**LLM 全程摸不到**。

状态机（5 态）：`pending → in_progress → completed`，`in_progress → failed`（3 次全败），`pending → canceled`（用户取消）。

### 2.2 Plan = 工单本 + 仓库（`models/plan.py`）

```python
class Plan(BaseModel):
    id: str                        # = checkpoint 的 thread_id
    status: PlanStatus             # pending/in_progress/completed/failed/canceled
    steps: list[Step]              # ⭐ 有序步骤数组：下标即依赖即执行序
    artifacts: dict[str, Artifact] # ⭐ 全局产物仓库（Agent 间消息总线）
    user_intent / intent_type / trace_id / created_at / updated_at
```

**白话**：Plan 出生在 planner_node，之后住进 `state["plan"]`，图节点原地改它、改完随返回值落 checkpoint。瘦掉的字段：`edges`（边即下标）、`budget`（被"3 次重试 + fail-fast"替代）、`meta`（进度由 steps 状态推导）。

### 2.3 Artifact = 交接棒（`models/artifact.py`）

```python
class Artifact(BaseModel):
    key: str               # "parsed_paper" / "repo_url" / "run_result" / "final_report"
    type: ArtifactType     # json/code/url/text/metrics/report/...
    producer_task_id: str  # 谁产的（溯源）
    value: Any             # 内容
    metadata: dict         # {"ok": True, "attempts": 1, ...}
```

**三个结构的血缘**：Plan 装着一堆 Step；每个 Step 完成后把货（Artifact）写进 `Plan.artifacts`；下游 Step 的 `required_artifacts` 写明要领哪个 key 的货。**state 里没有 events 字段**——事件走 event_sink 旁路直推 SSE，不进 state。

### 2.4 AgentState（`state.py`）

```python
class AgentState(TypedDict):
    user_input: str
    session_id: str
    plan_id: str
    intent: IntentContext
    plan: Plan
    artifacts: dict[str, Artifact]
    final_report: str
    error: str | None
    current_step: int                            # ⭐ 游标：执行到第几步
    revision_count: Annotated[int, operator.add] # reducer 累加：revise 计数
```

原"执行循环账本"（current_task / attempt_counts / total_attempts）已删——重试计数收敛进 `_run_step_with_retry` 的局部变量，游标 `current_step` 是唯一的全局执行进度。

---

## 3. 一次请求的完整旅程（抽丝剥茧）

以"复现 Attention Is All You Need 论文"为例，跟着数据走一遍。

### 站 0：POST /runs（API 层，`routes.py`）

```python
plan_id = str(uuid.uuid4())
state.bus.subscribe(plan_id)    # 先订阅 SSE（幂等，队列缓冲）
state.active_runs.add(plan_id)  # 占位（防 SSE 竞态 404）
state.spawn(_run_full_flow(state, plan_id, user_input, session_id))
```

**细节**：plan_id 由 API 层预生成而不是等 planner——SSE 队列必须先于图执行订阅，而 plan 要等 planner_node 才诞生，所以 ID 反过来传进图里。返回 202 后前端立即连 `/stream`。

**📂 涉及文件**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `api/routes.py` | 端点定义 + 后台任务体 | `run_one_shot` / `_run_full_flow` / `_thread_config` |
| `api/app_state.py` | 依赖注入容器（bus / 并发占位 / 任务强引用） | `AppState.spawn` / `active_runs` |
| `api/sse.py` | SSE 事件总线（幂等订阅 + 队列缓冲 + 15s 心跳） | `EventBus.subscribe/publish` / `event_stream` |
| `models/event.py` | 事件模型 | `PlanEvent` / `PlanEventType` |
| `main.py` | 装配入口（lifespan 建依赖 + include_router 挂路由） | `build_app_state` |

### 站 1：intent_node——意图识别（三路并行）

```
用户输入 → rule_router 预分类（关键词命中？直接返回，0 次 LLM）
        → 没命中 → Redis 记忆加载 → 三路并行调 LLM
```

三路并行（`asyncio.gather(return_exceptions=True)`，对应 Go 的 errgroup）：

| 路 | 任务 | 失败语义 |
|---|---|---|
| A | ClassifyOnly：意图类型 + 实体 + 置信度 | **致命** → unknown → END |
| B | Rewrite：query 重写为专业表述 | 非致命 → 降级用原 query |
| C | ExtractPaperFields：paper_title / arxiv_id | 非致命 → 降级空 dict |

三出口全用 `Command` 导航：
- 意图不合格/识别异常 → END（补发 plan_failed 收 SSE 流）
- 意图合格 → planner
- `state["plan"]` 已存在 → 直进 execute_step（图级直连入口，测试/内部复用）

**为什么三路并行**：三次 LLM 调用相互独立，串行要 3 倍延迟。**为什么独立降级**：重写失败不该连累分类结果，抽取失败不该连累重写——`return_exceptions=True` 让异常作为结果返回，调用方逐路判断。

**📂 涉及文件**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `graph.py` | intent 节点（Command 三出口导航） | `intent_node` |
| `agent/intent/classifier.py` | 意图识别主入口：规则预分类 → 三路并行 → 独立降级 | `IntentClassifier.classify` / `_classify_only` / `_rewrite` / `_extract_paper_fields` |
| `agent/intent/rule_router.py` | 规则快速路径（关键词命中直接返回，0 次 LLM） | `route_by_rule` |
| `agent/intent/memory.py` | Redis 意图记忆（多轮上下文，降级 Noop） | `IntentMemoryStore.load_recent_turns` / `append_turn` |
| `agent/prompts/intent_prompts.py` | 三路各自的 system/user prompt | `CLASSIFY_SYSTEM` / `REWRITE_SYSTEM` / `EXTRACT_SYSTEM` |
| `models/intent.py` | 意图模型（三路结果汇成它） | `IntentContext` / `IntentType` / `PaperSearchFields` |
| `agent/llm_client.py` | LLM 底层封装（三路共用） | `LLMClient.ainvoke` / `with_structured_output` |

### 站 2：planner_node——规划（LLM 优先 + 两道闸 + 模板兜底）

```
LLM 规划（Structured Outputs 出 PlanBlueprint）
  → 闸1 validate_critical_contracts：高契约步骤必须派给确定性 Agent
  → 闸2 validate_steps：产物契约闭合（producer 存在 + 唯一）
  → 任一不过 → 模板兜底（确定性，永远可用）
```

**闸 1 高契约**（`validator.py`）：

```python
CRITICAL_CONTRACTS = {"repo_discovery": "research_coding_agent"}
```

repo_discovery 涉及外部 URL 选择——LLM 会编造 URL 或选错文件，破坏可重复性。LLM 把它派给别的 Agent？整份规划作废，回退模板。

**闸 2 产物契约**：每个 `required_artifacts` 必须有 producer（防消费不存在的产物），每个产物只能有一个 producer（防写冲突）。

**原五项校验为什么只剩两项**："依赖节点存在"和"Kahn 检环"已删——步骤数组在结构上不可能引用不存在的依赖（下标即依赖），也不可能成环（下标单向递增）。**这类错误由数据结构本身杜绝，比运行时校验更可靠。**

**📂 涉及文件**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `graph.py` | planner 节点（调 build_plan + 初始化执行游标） | `planner_node` |
| `agent/planner/planner.py` | 规划主入口：LLM 优先 → 两道闸 → 模板兜底 | `build_plan` / `_try_llm_plan` |
| `agent/planner/agent_planner.py` | 调 LLM 出结构化蓝图（Structured Outputs 强制 schema，无 dependencies 字段） | `build_steps` / `PlanBlueprint` / `_extract_step_inputs` |
| `agent/planner/validator.py` | 两道闸（约束 LLM 的核心：高契约派工 + 产物契约闭合） | `validate_critical_contracts` / `validate_steps` |
| `agent/planner/templates.py` | 4 类意图的确定性步骤模板（永远可用的兜底） | `build_steps_from_template` / `TEMPLATES` |
| `agent/prompts/planner_prompts.py` | 规划 prompt（LLM 的完整 intent dump） | `PLANNER_SYSTEM` / `planner_user_prompt` |
| `models/step.py` + `models/plan.py` | 规划的输出模型 | `Step` / `Plan` |

### 站 3：approval_node——人工审批（interrupt 挂起）

```python
async def approval_node(state: AgentState) -> Command:
    decision = interrupt({          # ⭐ 整个节点体只有这一件事
        "plan": state["plan"].model_dump(mode="json"),
        "revision_count": state.get("revision_count") or 0,
    })
    action = decision.get("action", "")
    if action == "approve":
        return Command(goto="execute_step")
    if action == "revise":
        # feedback 并入 intent 回炉重新规划
        ...
        return Command(goto="planner", update={..., "revision_count": 1})
    # abandon：终态 canceled
    return Command(goto="report", update={"plan": plan})
```

**三个关键设计**：

1. **节点体只有 interrupt() 一件事**：resume 会从挂起节点第一行完整重跑——interrupt 之前不能有任何副作用（不调 LLM、不发事件、不改 state），否则恢复时全部重复执行。`plan_awaiting_approval` 事件由 API 层在 ainvoke 返回后补发（唯一出口，绝不重复）。

2. **interrupt payload 随 checkpoint 持久化**：审批请求（plan 全量 + 已修订次数）重启不丢，`GET /plans/{id}` 随时查回。这是"状态管理只剩一套"的根源——没有内存 registry，没有两段式 REST。

3. **revise 的 feedback 并入 `intent.raw_intent`**：planner 的 LLM prompt 是完整 intent dump，只写 user_input 的话重规划根本看不到修改意见——追加到原话后面（原话留史、实体/约束保留），新计划才能吸收反馈。修订上限 MAX_REVISIONS=3 拦在 API 层（调用侧守门），图内 revision_count 只如实记账（reducer 累加）——职责分工。

**📂 涉及文件**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `graph.py` | 审批节点——**节点体只有 interrupt() 一件事**（不调 LLM、不发事件、不改 state） | `approval_node` |

> 全项目"最省"的节点：没有调用任何业务文件。interrupt payload（plan 全量 + revision_count）随 checkpoint 持久化；resume 唤醒后按 resume 值驱动 Command 分流。补发 `plan_awaiting_approval` 事件在 API 层（`routes.py` 的 `_emit_awaiting_if_suspended`），因为它必须在 ainvoke 返回后检查快照才能发，且是幂等唯一出口。

### 站 4：execute_step——执行循环（自环 + 游标）

```python
async def execute_step_node(state: AgentState) -> Command:
    plan = state["plan"]
    idx = state.get("current_step") or 0

    if plan.id in canceled:                     # 协作式取消
        _mark_remaining_canceled(plan, idx)
        return Command(goto="report", update={"plan": plan})

    if idx >= len(plan.steps):                  # 全部跑完
        return Command(goto="report", update={"plan": plan})

    step = plan.steps[idx]
    ok = await _run_step_with_retry(plan, step) # 3 次原地重试
    if not ok:
        return Command(goto="report", update={"plan": plan})  # fail-fast

    return Command(goto="execute_step", update={  # ⭐ 自环：游标+1
        "plan": plan, "artifacts": plan.artifacts, "current_step": idx + 1,
    })
```

**四条铁律**：

1. **游标唯一推进点在成功 return 里**——落盘与完成同一瞬间，游标不可能停在"做了一半"的位置。断点续跑粒度 = 步：每完成一步 return 一次，plan + 游标随 checkpoint 原子落盘，崩溃后恢复已完成步骤不重跑。

2. **artifacts 必须随 update 同步**——state 字段是 checkpoint 入账的唯一通道，`plan.artifacts` 的原地累积不会自动反映到 `state["artifacts"]`（这是踩过的一个坑：report 读 state["artifacts"] 拿到的是空 dict）。

3. **单步 3 次重试 + fail-fast**：防偶发网络/超时抖动重跑一次基本就好；3 次全败说明是必死任务（repo 404 / 论文太冷门），继续重试只会烧钱。链式下下游全依赖本步，跑下去也必然失败——下游保持 pending 原样留给用户看，不再有 blocked 连坐。

4. **未注册 Agent 是确定性错误**（重试无意义，1 次都不多跑）——`workers.get(step.agent)` 查不到直接终局失败。

**协作式取消**：不硬杀正在跑的活，`canceled` 是闭包里的一个 set，`/cancel` 端点往里塞 plan_id，execute_step 每步开头看一眼——"当前步骤跑到哪算哪"语义，已花的算力不浪费。

**📂 涉及文件**——本节点是"调度层 + 执行层"两层，文件最多：

**① 调度层（graph.py）**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `graph.py` | 执行循环：按下标取步骤 → 派工 → 重试 → 收尾 | `execute_step_node` / `_run_step_with_retry` / `_finalize_success` / `_mark_remaining_canceled` |

**② 执行层（4 个 worker，LLM 真正干活的地方）**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `agent/workers/base.py` | Agent 统一接口契约 `execute(step, artifacts) → dict[Artifact]` | `BaseAgent.execute` |
| `agent/workers/librarian.py` | 解析论文（纯 LLM） | `LibrarianAgent`（paper_parse → parsed_paper） |
| `agent/workers/coder.py` | 生成/执行代码（LLM + 沙箱） | `CoderAgent`（code_generate / code_run / framework_compare） |
| `agent/workers/research_coding.py` | 搜 repo / 跑仓库（**确定性工具链 + Harness**，LLM 只在修复阶段） | `ResearchCodingAgent`（repo_discovery / code_run） |
| `agent/workers/data.py` | 汇总 artifacts 生成报告（纯 LLM） | `DataAgent`（report → final_report） |
| `agent/prompts/worker_prompts.py` | 四个 worker 的 system/user prompt | `LIBRARIAN_SYSTEM` / `CODER_SYSTEM` / `DATA_SYSTEM` |
| `models/artifact.py` | 交接棒（worker 的返回值形状） | `Artifact` / `ArtifactType` |

**③ ResearchCoding 内部的确定性工具链**（repo_discovery / code_run 步骤会用到）：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `agent/tools/github_search.py` | GitHub 搜索 + star/活跃度质量过滤 | `search_repos` / `pick_best_repo` |
| `agent/tools/dependency_installer.py` | 依赖安装（联网阶段）+ 失败降级 | `install_dependencies` |
| `agent/sandbox/docker_executor.py` | 两阶段网络沙箱：装依赖联网 → 执行断网 | `create_for_install` / `switch_to_run_phase` / `execute` / `cleanup` |
| `agent/sandbox/registry.py` | 容器索引（Redis，崩溃恢复用） | `Registry` |
| `agent/sandbox/security.py` | 资源限制（CPU/内存/超时） | 限额配置 |

**④ Harness 自愈引擎**（被 research_coding 的 code_run 调用，5 道闸）：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `agent/harness/runner.py` | 自愈主循环（1..3 轮 + 五道闸 + 负反馈） | `Harness.run_with_healing` |
| `agent/harness/patch_policy.py` | 闸1：补丁静态校验（防越权） | `validate_patch` |
| `agent/harness/fingerprint.py` | 闸3：仓库指纹（防执行期偷改文件） | `compute_fingerprint` |
| `agent/harness/metrics_recompute.py` | 闸4：指标重算（防造假） | `recompute_metrics` |
| `agent/harness/circuit_breaker.py` | 闸5：熔断器（防死循环修复烧 token） | `CircuitBreaker` |
| `agent/harness/feedback_injector.py` | 历史负反馈注入修复 prompt | `build_feedback_prompt` |

> 阅读建议：先看 ① 的 `_run_step_with_retry`（怎么派工/重试），再按 step.type 跳到 ② 对应的 worker；`repo_discovery` / `code_run` 步骤再深入 ③④。

### 站 5：report_node——收尾

终态判定优先级：取消 > 全部成功 > 失败（`_decide_terminal_status`）。canceled 和 completed 一样不算错误（用户主动选择）。`final_report` 缺失时拼装降级摘要（`_build_fallback_report`：步骤状态清单 + 产物截断展示）。

**📂 涉及文件**：

| 文件 | 该站职责 | 关键符号 |
|---|---|---|
| `graph.py` | 终态判定 + 发终态事件 + 拼最终报告 | `report_node` / `_decide_terminal_status` / `_build_fallback_report` |

> 注意区分：**report_node（图节点）≠ DataAgent（report 步骤的 worker）**。正常流程里 final_report 由 DataAgent 在 execute_step 的 report 步骤生成（见站 4 ②），report_node 只是从 `artifacts["final_report"]` 取出来做终态收尾；只有 report 步骤失败/被跳过时，report_node 才用 `_build_fallback_report` 现场拼降级摘要。图节点和 worker 是两层，别混。

---

## 4. 亮点一：代码约束 LLM（LLM 触点清单）

这是项目的第一设计原则。执行层的每个决策点，问一句"这是 LLM 说了算还是代码说了算"：

| 决策 | 谁说了算 | 怎么约束 |
|---|---|---|
| 出什么步骤 | LLM 提议 → **代码裁决** | validator 两道闸（高契约派工 + 产物契约闭合），不过则模板兜底 |
| 先跑哪步 | **代码** | 数组下标（数学性质，不是 LLM"觉得"） |
| 派给谁 | LLM 填字段 → **代码查表** | workers 名册路由；高契约步骤（repo_discovery）必须归确定性 Agent |
| 要不要重试 | **代码** | MAX_STEP_ATTEMPTS=3 计数器 |
| 代码挂了怎么修 | LLM 提议 → **代码验收** | Harness 5 道闸（见亮点三） |
| 跑没跑成 | **代码** | exit_code / 指纹 / 指标重算 |
| 计划要不要执行 | **人** | interrupt() 挂起，approve/revise/abandon |

**LLM 在执行层的触点为零**。LLM 的价值被精确限定在"创造性建议"（怎么规划、怎么修代码、怎么写报告），所有"确定性裁决"（顺序、路由、重试、验收）都在代码手里。

---

## 5. 亮点二：多 Agent + 意图识别

### 5.1 四个异构 Agent

| Agent | 能力构成 | 步骤类型 | LLM 用法 |
|---|---|---|---|
| LibrarianAgent | 纯 LLM | paper_parse | 生成论文解析 |
| CoderAgent | LLM + 沙箱 | code_generate / code_run / framework_compare | 生成代码，执行在沙箱 |
| ResearchCodingAgent | **确定性工具链 + Harness** | repo_discovery / code_run | 只在 Harness 修复阶段用 |
| DataAgent | 纯 LLM | report | 汇总 artifacts 出报告 |

**多 agent 的本质不在节点数量，在三点**：
1. **异构能力**：四个 Agent 的能力构成不同（谁调 LLM、谁碰沙箱、谁纯确定性）；
2. **动态编排**：LLM 每次请求现编步骤（选谁、几步、什么序），validator 校验后生效；
3. **消息总线**：artifacts 全局仓库做 Agent 间通信——生产者写、消费者按 required_artifacts 取，解耦且可溯源（producer_task_id）。

这正是 LangGraph 官方 plan-and-execute 模式的落地：planner 出计划、executor 循环执行、artifacts 传数据。

### 5.2 意图识别为什么是亮点

不只是"分类"——它是**三路并行的理解管线**：

```
"帮我复现下 attention 那篇论文"
  ├─ 路A 分类：intent_type=Paper_Reproduction, entities={...}, confidence=0.9
  ├─ 路B 重写："复现 Attention Is All You Need 论文"（专业表述，利于后续检索）
  └─ 路C 抽取：paper_title="Attention Is All You Need"（喂给 repo 搜索）
```

加上**规则快速路径**（关键词命中直接返回，0 次 LLM）、**Redis 会话记忆**（多轮上下文，降级 Noop 不影响主流程）、**独立降级语义**（A 致命、B/C 非致命），构成了一个生产级的意图理解子系统。

---

## 6. 亮点三：Sandbox + Harness 自愈

### 6.1 Docker 两阶段网络沙箱（`sandbox/docker_executor.py`）

```
阶段1 create_for_install：bridge 网络容器（联网）
    └─ pip install numpy torch ...（需要访问 PyPI）
阶段2 switch_to_run_phase：断网容器（workspace 卷保留依赖）
    └─ python run.py（隔离执行，防恶意代码外传数据）
```

**为什么两阶段**：装依赖必须联网，跑代码必须断网——安全性和可用性各占一头。切换时容器重建但 workspace 卷不动，依赖不丢。

配套：registry（Redis 容器索引，崩溃恢复用）、security（CPU/内存/超时限制）、clone 三策略降级（shallow → full → codeload tar.gz）。

### 6.2 Harness 自愈引擎（`harness/runner.py`，项目灵魂）

**场景**：复现仓库的代码跑挂了（ImportError、路径错、版本不兼容）。传统做法要么放弃要么人肉修。Harness 让 LLM 修，但**每次修复都要过 5 道闸**：

```python
for attempt in 1..3:
    if attempt > 1 and 代码 SHA256 没变: continue          # 闸2a：假修复检测
    if 补丁不合法（Patch Policy）: LLM 修复 + continue      # 闸1：越权防护
    备份 + 写入 + 执行
    if 成功:
        if 执行期间偷改文件（Fingerprint）: LLM 修复 + continue  # 闸3：越权写入
        if 指标造假（重算不匹配）: LLM 修复 + continue           # 闸4：造假检测
        还原 + return 成功
    if 熔断（CircuitBreaker）: return 失败                  # 闸5：止损
    注入负反馈（FeedbackInjector）+ LLM 修复                # 机制：负反馈学习
```

五道闸各自防什么：

| 闸 | 机制 | 防的坏行为 |
|---|---|---|
| 1 | patch_policy 静态校验 | LLM 写 `os.system("rm -rf /")` 之类的越权补丁 |
| 2 | SHA256 前后对比 | LLM 假装修了（返回原代码），白烧一轮执行 |
| 3 | 仓库指纹（全部文件 hash） | 代码执行期间偷改源文件（比如改掉评估逻辑） |
| 4 | 指标重算（metrics.json vs predictions.jsonl 重算） | 跑通了但指标造假（硬编码 accuracy=0.99） |
| 5 | 熔断器（window=2 连续同因失败） | 死循环修复同一类错误烧 token |

**关键设计点**：
- **复用持久化沙箱**（sandbox_id + exec_in），不起新容器——前面 clone + 装依赖的成果不能丢；
- **入口文件加白名单**：Harness 修复的就是入口文件，指纹对比时把它排除，否则自己修自己报；
- **负反馈注入**：修复 prompt 里带上历史失败记录（哪次、什么错、exit_code），LLM 不会在同一个坑里摔三次；
- **备份/还原**：每轮执行前备份入口文件，失败后还原，保证下轮从干净状态开始。

---

## 7. API 契约

### 7.1 端点一览

| 端点 | 方法 | 语义 | 关键状态码 |
|---|---|---|---|
| `/api/v1/runs` | POST | 一段式起图（意图→规划→审批挂起） | 202 |
| `/api/v1/plans/{id}/resume` | POST | 审批决定（approve/revise/abandon） | 202 / 404 / 409 / 422 |
| `/api/v1/plans/{id}` | GET | 查状态（checkpoint 权威） | 200 / 404 |
| `/api/v1/plans/{id}/stream` | GET | SSE 事件流（15s 心跳） | 200 / 404 |
| `/api/v1/plans/{id}/cancel` | POST | 取消（挂起转 abandon / 执行中软取消） | 200 / 404 / 409 |

### 7.2 状态唯一权威 = checkpoint

`thread_id = plan_id`。plan 的生命周期状态全部住在 LangGraph checkpoint 里——GET / resume / cancel 都从 `aget_state` 读权威状态。挂起判定：`"approval" in snapshot.next`。

**409 的两种语义**：
- resume 时不在挂起态（终态后重复 approve）
- resume 处理中（resuming_runs 占位，双击守卫）

**并发控制**：`active_runs`（初始 run）和 `resuming_runs`（resume 操作）分开——图挂起后初始任务还在收尾（发事件/清标记），此时 resume 完全合法，不能误伤。

### 7.3 SSE 事件流

```
plan_awaiting_approval   → task_started → task_completed → artifact_created
→ task_retrying → task_failed → plan_completed / plan_failed / plan_canceled
```

挂起期间可能长时间无事件——15s 心跳保活。事件走 EventBus 旁路（`graph._emit` → `bus.sink`），不进 state；sink 挂了只 warning 不影响主流程。

---

## 8. 测试与运行

### 8.1 测试全景（106 个，全绿）

| 文件 | 数量 | 测什么 |
|---|---|---|
| test_api.py | 19 | HTTP 层：审批三动作、双击守卫、修订上限、取消三分支、SSE |
| test_e2e.py | 2 | 全链路：真 workers + 假外部边界（LLM/GitHub/clone/Docker） |
| test_harness.py | 20 | 自愈引擎：五道闸各自触发场景 |
| test_service_graph.py | 11 | 图编排：自环推进、重试、fail-fast、取消、interrupt 挂起/恢复 |
| test_planner.py | 10 | 两道闸 + LLM 优先/兜底 |
| test_workers.py | 19 | 四个 Agent 的分发逻辑 + 产物契约 |
| test_sandbox.py | 10 | 沙箱生命周期 |
| test_intent_classifier.py | 5 | 三路并行 + 降级 |
| test_tools.py | 8 | GitHub 搜索 / 依赖安装 |
| test_models.py | 2 | 模型默认值 + 序列化往返 |

**测试原则**：不联网、不花钱、不起真容器。FakeLLM（预设响应按序弹出）+ fakeredis + StubAgent / FakeSandbox（脚本化返回值）+ monkeypatch 网络边界。

### 8.2 运行

```bash
# 安装
pip install -e ".[dev]"
# 或（与 CI 一致，uv 版）：uv sync --extra dev

# 测试
pytest tests/ -q
# 或（与 CI 一致）：uv run pytest

# 规则快速路径样本覆盖速测（30 条典型指令，实测命中 77%）
uv run python scripts/rule_router_sample.py

# CI：push/PR 自动跑全套（GitHub Actions；106 条全离线，无需任何密钥）

# 起服务（需要 .env 配 DEEPSEEK_API_KEY；Docker daemon 运行中）
uvicorn scholar_agent.main:app --reload

# 体验一次
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"user_input": "复现 Attention Is All You Need 论文"}'
# → {"plan_id": "...", "status": "in_progress"}
# 浏览器开 SSE：GET /api/v1/plans/{plan_id}/stream
# 审批：POST /api/v1/plans/{plan_id}/resume -d '{"action": "approve"}'
```

---

## 9. 面试速答（把项目讲清楚的最短路径）

**Q: 这项目解决什么问题？**
论文复现自动化：用户给一句话，系统识别意图、规划步骤、搜 GitHub 复现仓库、在 Docker 沙箱里跑、代码挂了 LLM 自动修（5 道闸验收），全程 SSE 直播，执行前人工审批。

**Q: 为什么用步骤数组不用 DAG？**
业务图恒为链式（解析→搜repo→跑代码→报告），链是 DAG 的特例——数组下标同时承载拓扑序、依赖序、执行序。通用 DAG 调度器（拓扑排序/叫号/连坐）对这个业务是纯开销。砍掉后图节点 7→5，调度代码 300 行→60 行。

**Q: 怎么保证 LLM 不乱来？**
设计原则"模型负责建议，代码负责约束"。LLM 的触点被精确限定：规划提议（两道闸校验+模板兜底）、代码修复（5 道闸验收）、内容生成（纯产出无决策）。执行顺序、Agent 路由、重试、成败判定全部确定性代码。

**Q: Harness 自愈是怎么工作的？**
3 次重试循环，每轮：补丁静态校验（防越权）→ SHA256 对比（防假修复）→ 执行 → 指纹校验（防偷改文件）→ 指标重算（防造假）→ 熔断检测（防死循环）。修复 prompt 注入历史负反馈，LLM 不会重复踩坑。

**Q: 人工审批怎么实现的？**
LangGraph 原生 interrupt()：approval 节点调用后整图挂起，审批 payload 随 checkpoint 持久化（重启不丢）。用户调 /resume 传 Command(resume=决定) 唤醒。节点体只有 interrupt() 一件事——resume 会从节点第一行重跑，之前不能有副作用。

**Q: 多 agent 体现在哪？**
4 个异构 Agent（纯 LLM / LLM+沙箱 / 确定性工具链 / 纯 LLM）+ LLM 动态编排步骤 + artifacts 全局仓库做消息总线。不是"多个节点"就是多 agent，是"异构能力 + 动态编排 + 消息总线"三件套。
