"""端到端测试：HTTP 全链路（mock 掉外部边界，业务代码全真）。

与单元测试的区别：
    - 单测（test_api.py）：graph 里塞 StubAgent，测的是"调度正确"
    - E2E（本文件）：build_app_state 组装【真 workers + 真 services + 真 Harness】，
      只 mock 四个外部边界——LLM（FakeLLM 脚本）、GitHub 搜索、git clone、Docker（本地 subprocess 替代）

验证的完整链路（interrupt 审批模式）：
    HTTP POST /runs → 意图识别（规则）→ 模板规划 → approval 挂起
    → POST /resume(approve) → 真四类 Agent 按拓扑序执行 → 依赖安装
    → Harness 自愈执行 → 事件经 EventBus → GET /stream SSE
    → GET /plans/{id} 终态

场景：
    1. 快乐路径：论文复现 4 步骤全绿 → plan_completed
    2. 降级路径：repo 搜不到 → s2 重试耗尽失败 → fail-fast
       （s3/s4 保持 pending 不执行）→ plan_failed
"""

import asyncio
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import fakeredis.aioredis
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from scholar_agent.agent.sandbox.result import SandboxResult
from scholar_agent.agent.workers.research_coding import ResearchCodingAgent
from scholar_agent.api.app_state import build_app_state
from scholar_agent.api.routes import get_state
from scholar_agent.main import app
from tests.conftest import FakeLLM

# ──────────────────────────────────────────────
# 外部边界 1+4：Fake 持久化沙箱（本地 subprocess 替代 Docker）
# ──────────────────────────────────────────────


class FakeRegistry:
    """sandbox.registry 替身（容器索引不落 Redis）。"""

    async def remove(self, plan_id: str) -> None:
        pass


class FakePersistentSandbox:
    """两阶段持久化沙箱替身：python 命令本地真跑，其余命令假装成功。

    这样 Harness 的自愈循环、指纹校验、入口执行全部走真代码路径，
    只是"容器"换成了本地 subprocess。
    """

    def __init__(self):
        self.workspaces: dict[str, str] = {}
        self.commands: list[list[str]] = []
        self.registry = FakeRegistry()

    async def create_for_install(self, workspace: str, plan_id: str, image: str = "") -> str:
        cid = f"fake-install-{plan_id[:8]}"
        self.workspaces[cid] = workspace
        return cid

    async def switch_to_run_phase(self, container_id: str) -> str:
        workspace = self.workspaces.pop(container_id)
        cid = f"fake-run-{len(self.workspaces)}"
        self.workspaces[cid] = workspace
        return cid

    async def exec_in(self, container_id: str, command: list[str]) -> SandboxResult:
        self.commands.append(command)
        workspace = self.workspaces.get(container_id)
        if command[:1] == ["python"]:
            proc = subprocess.run(  # noqa: S603 本地跑自建的 run.py，无外部输入
                command, cwd=workspace, capture_output=True, text=True, timeout=60,
            )
            return SandboxResult(
                stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode,
            )
        return SandboxResult(stdout="ok", exit_code=0)  # pip install 等

    async def cleanup(self, container_id: str) -> None:
        self.workspaces.pop(container_id, None)


# ──────────────────────────────────────────────
# 外部边界 2+3：GitHub 搜索 / git clone 的 monkeypatch
# ──────────────────────────────────────────────

FAKE_REPO = {
    "full_name": "fake/attention-impl",
    "html_url": "https://github.com/fake/attention-impl",
    "stars": 1200,
    "pushed_at": datetime.now(UTC).isoformat(),
}


def patch_search_ok(monkeypatch):
    """GitHub 搜索 → 返回 1 个高星活跃候选。注意：真 search_repos 是同步函数
    （worker 里用 asyncio.to_thread 调），fake 也必须同步。"""
    def fake_search(query, token=None, max_candidates=5):
        return [FAKE_REPO]

    # research_coding 模块内 from-import 了 search_repos，patch 模块属性
    monkeypatch.setattr(
        "scholar_agent.agent.workers.research_coding.search_repos", fake_search
    )


def patch_search_empty(monkeypatch):
    """GitHub 搜索 → 无候选（触发 repo_discovery 失败场景）。"""
    def fake_search(query, token=None, max_candidates=5):
        return []

    monkeypatch.setattr(
        "scholar_agent.agent.workers.research_coding.search_repos", fake_search
    )


def patch_clone(monkeypatch):
    """git clone → 本地伪造 workspace（requirements.txt + run.py）。"""
    async def fake_clone(self, repo_url: str) -> str:
        workspace = tempfile.mkdtemp(prefix="e2e_repo_")
        Path(workspace, "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
        Path(workspace, "run.py").write_text(
            'print("accuracy=0.95")\n', encoding="utf-8",
        )
        return workspace

    monkeypatch.setattr(ResearchCodingAgent, "_clone_and_setup", fake_clone)


# ──────────────────────────────────────────────
# 组装：真 workers + 假外部边界
# ──────────────────────────────────────────────


def make_e2e_state(llm_responses: list[str]):
    """build_app_state 组装全真业务层（区别于 test_api 的 StubAgent 拼装）。

    checkpointer 用 InMemorySaver：approval 的 interrupt 挂起必备底座
    （生产由 main._build_checkpointer 提供 Redis 版）。
    """
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    sandbox = FakePersistentSandbox()
    state = build_app_state(
        FakeLLM(responses=llm_responses), sandbox, redis,
        checkpointer=InMemorySaver(),
    )
    return state, sandbox


async def run_and_approve(client) -> str:
    """/runs 起图（跑到 approval 挂起）+ approve 放行，返回 plan_id。"""
    resp = await client.post(
        "/api/v1/runs", json={"user_input": "复现 Attention Is All You Need"}
    )
    assert resp.status_code == 202
    plan_id = resp.json()["plan_id"]

    # 轮询等 approval 挂起（图停在 interrupt，快照可查）
    for _ in range(200):
        plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
        if plan.get("awaiting_approval"):
            break
        await asyncio.sleep(0.05)
    else:
        raise TimeoutError("图未挂起到审批点")

    resp = await client.post(
        f"/api/v1/plans/{plan_id}/resume", json={"action": "approve"}
    )
    assert resp.status_code == 202
    return plan_id


async def wait_terminal(client, plan_id, timeout=30) -> dict:
    """轮询到终态（真 Harness 执行较慢，timeout 放宽）。"""
    for _ in range(timeout * 20):
        plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
        if plan["status"] in ("completed", "failed", "canceled"):
            return plan
        await asyncio.sleep(0.05)
    raise TimeoutError(f"plan {plan_id} 未到终态: {plan['status']}")


def sse_event_types(text: str) -> list[str]:
    return [
        line.removeprefix("event: ")
        for line in text.splitlines()
        if line.startswith("event: ")
    ]


# ──────────────────────────────────────────────
# 场景 1：快乐路径（4 节点全绿）
# ──────────────────────────────────────────────


async def test_e2e_paper_reproduction_happy_path(monkeypatch):
    """HTTP → 意图 → 模板 → approval 挂起 → approve → 真 workers
    （搜repo/装依赖/Harness）→ SSE → 终态 completed。

    FakeLLM 脚本（按 ainvoke 顺序）：
        1. librarian 解析论文
        2. data 生成报告
    （repo_discovery / clone / 安装 / Harness 首轮通过，均不需要 LLM）
    """
    patch_search_ok(monkeypatch)
    patch_clone(monkeypatch)
    state, sandbox = make_e2e_state([
        "论文解析：Transformer，自注意力机制……",   # n1 librarian
        "# 复现报告\n复现成功，指标对齐。",          # n4 data report
    ])
    app.dependency_overrides[get_state] = lambda: state
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            plan_id = await run_and_approve(client)
            await wait_terminal(client, plan_id)

            # SSE 全链路事件（含审批挂起事件，全缓冲在队列里）
            stream = await client.get(f"/api/v1/plans/{plan_id}/stream")
            events = sse_event_types(stream.text)
            assert events[0] == "plan_awaiting_approval"  # 先挂起送审
            assert events.count("task_started") >= 4
            assert events.count("task_completed") == 4
            assert events[-1] == "plan_completed"

            # 终态与产物
            plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
            assert plan["status"] == "completed"
            assert plan["awaiting_approval"] is False
            assert [s["status"] for s in plan["steps"]] == ["completed"] * 4
            artifacts = plan["artifacts"]
            assert set(artifacts) == {
                "parsed_paper", "repo_url", "run_result", "final_report",
            }
            # repo 来自伪造的 GitHub 候选（URL 校验真实跑过）
            assert artifacts["repo_url"]["value"] == FAKE_REPO["html_url"]
            # Harness 在"本地容器"里真跑了 run.py，stdout 进了 run_result
            assert "accuracy=0.95" in artifacts["run_result"]["value"]
            assert artifacts["run_result"]["metadata"]["ok"] is True
            # 依赖安装命令真的发过（带 pip index）
            assert any(c[:2] == ["pip", "install"] for c in sandbox.commands)
    finally:
        app.dependency_overrides.clear()


# ──────────────────────────────────────────────
# 场景 2：降级路径（repo 搜不到 → fail-fast）
# ──────────────────────────────────────────────


async def test_e2e_repo_not_found_blocks_downstream(monkeypatch):
    """搜索无候选 → s2 重试 3 次耗尽失败 → fail-fast 进 report
    （链式下 s3/s4 全依赖 s2，跑下去也必然失败——保持 pending
    不烧冤枉钱）→ plan_failed。

    验证重试事件、fail-fast 语义、终态事件，全部经 SSE 可见
    （审批挂起 → approve 放行后才进执行循环）。
    """
    patch_search_empty(monkeypatch)
    patch_clone(monkeypatch)  # n3 不会执行到 clone，patch 了也无妨
    state, _sandbox = make_e2e_state([
        "论文解析：Transformer……",  # 只有 n1 会调 LLM
    ])
    app.dependency_overrides[get_state] = lambda: state
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            plan_id = await run_and_approve(client)
            await wait_terminal(client, plan_id)

            stream = await client.get(f"/api/v1/plans/{plan_id}/stream")
            events = sse_event_types(stream.text)

            # s2 重试 2 次后第 3 次失败
            assert events.count("task_retrying") == 2
            assert "task_failed" in events
            # fail-fast：s3/s4 从未启动（只有 s1/s2 各 start 一次）
            assert events.count("task_started") == 2
            assert events[-1] == "plan_failed"

            plan = (await client.get(f"/api/v1/plans/{plan_id}")).json()
            assert plan["status"] == "failed"
            statuses = {s["id"]: s["status"] for s in plan["steps"]}
            assert statuses == {
                "s1": "completed", "s2": "failed",
                "s3": "pending", "s4": "pending",
            }
            # 产物只有 s1 的（失败计划不产报告）
            assert set(plan["artifacts"]) == {"parsed_paper"}
    finally:
        app.dependency_overrides.clear()
