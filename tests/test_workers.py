"""workers 模块功能测试：4 个 Agent 的分发逻辑 + 产物契约。

测试原则：FakeLLM / FakeSandbox 替身，monkeypatch 网络调用（GitHub 搜索、
git clone），不起真容器不联网。
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from tests.conftest import FakeLLM, make_step

from scholar_agent.agent.sandbox.result import SandboxResult
from scholar_agent.agent.tools.dependency_installer import RepairAction
from scholar_agent.agent.workers.coder import CoderAgent
from scholar_agent.agent.workers.data import DataAgent
from scholar_agent.agent.workers.librarian import LibrarianAgent
from scholar_agent.agent.workers.research_coding import ResearchCodingAgent
from scholar_agent.models.artifact import Artifact, ArtifactType

NOW = datetime.now(timezone.utc).isoformat()


def art(key: str, value, type_: ArtifactType = ArtifactType.text) -> Artifact:
    """快速构造 artifact。"""
    return Artifact(key=key, type=type_, value=value, producer_task_id="upstream")


class FakeOneShotSandbox:
    """CoderAgent 用：脚本化 execute() 返回值。"""

    def __init__(self, result: SandboxResult):
        self.result = result
        self.calls: list[str] = []

    async def execute(self, code: str, timeout: int = 300) -> SandboxResult:
        self.calls.append(code)
        return self.result


class FakeRegistry:
    async def remove(self, plan_id: str) -> None:
        self.removed = plan_id  # type: ignore[attr-defined]


class FakeRepoSandbox:
    """ResearchCodingAgent 用：脚本化持久化沙箱生命周期。"""

    def __init__(self, exec_results: list[SandboxResult]):
        self.exec_results = list(exec_results)
        self.registry = FakeRegistry()
        self.cleaned: list[str] = []

    async def create_for_install(self, workspace: str, plan_id: str, image: str = "") -> str:
        return "install-c1"

    async def switch_to_run_phase(self, container_id: str) -> str:
        return "run-c1"

    async def exec_in(self, container_id: str, command: list[str]) -> SandboxResult:
        if not self.exec_results:
            return SandboxResult(stdout="", exit_code=0)
        return self.exec_results.pop(0)

    async def cleanup(self, container_id: str) -> None:
        self.cleaned.append(container_id)


# ──────────────────────────────────────────────
# LibrarianAgent
# ──────────────────────────────────────────────


async def test_librarian_produces_parsed_paper():
    llm = FakeLLM(responses=["# 论文分析报告\n核心方法：Transformer..."])
    agent = LibrarianAgent(llm)  # type: ignore[arg-type]

    step = make_step("n1", type_="paper_parse", agent="librarian_agent",
                     inputs={"paper_title": "Attention Is All You Need"})
    out = await agent.execute(step, {})

    assert "parsed_paper" in out
    assert out["parsed_paper"].value.startswith("# 论文分析")
    assert out["parsed_paper"].producer_task_id == "n1"
    # system prompt 用的是 LIBRARIAN_SYSTEM
    assert "论文复现分析员" in llm.calls[0][0]["content"]


# ──────────────────────────────────────────────
# DataAgent
# ──────────────────────────────────────────────


async def test_data_collects_artifacts_into_report():
    llm = FakeLLM(responses=["# 最终报告\n复现成功"])
    agent = DataAgent(llm)  # type: ignore[arg-type]

    step = make_step("n3", type_="report", agent="data_agent")
    out = await agent.execute(step, {"run_result": art("run_result", "accuracy=0.75")})

    assert out["final_report"].type == ArtifactType.report
    assert "最终报告" in str(out["final_report"].value)
    # 上游产物内容进了 user prompt
    assert "accuracy=0.75" in llm.calls[0][1]["content"]


async def test_data_handles_empty_artifacts():
    llm = FakeLLM(responses=["报告：无输入"])
    agent = DataAgent(llm)  # type: ignore[arg-type]

    out = await agent.execute(make_step("n1", type_="report"), {})
    assert "final_report" in out


# ──────────────────────────────────────────────
# CoderAgent
# ──────────────────────────────────────────────


async def test_coder_generate_code():
    llm = FakeLLM(responses=["print('hello')"])
    agent = CoderAgent(llm, FakeOneShotSandbox(SandboxResult(stdout="", exit_code=0)))  # type: ignore[arg-type]

    step = make_step("n1", type_="code_generate")
    out = await agent.execute(step, {})

    assert out["generated_code"].type == ArtifactType.code
    assert out["generated_code"].value == "print('hello')"


async def test_coder_run_code_success():
    sandbox = FakeOneShotSandbox(SandboxResult(stdout="42", exit_code=0))
    agent = CoderAgent(FakeLLM(), sandbox)  # type: ignore[arg-type]

    step = make_step("n2", type_="code_run", required=["generated_code"])
    out = await agent.execute(step, {"generated_code": art("generated_code", "print(6*7)", ArtifactType.code)})

    assert out["run_result"].value == "42"
    assert out["run_result"].metadata["ok"] is True
    assert sandbox.calls == ["print(6*7)"]  # 执行的确实是上游生成的代码


async def test_coder_run_code_failure_records_stderr():
    sandbox = FakeOneShotSandbox(SandboxResult(stderr="NameError: x", exit_code=1))
    agent = CoderAgent(FakeLLM(), sandbox)  # type: ignore[arg-type]

    out = await agent.execute(
        make_step("n2", type_="code_run"),
        {"generated_code": art("generated_code", "bad", ArtifactType.code)},
    )
    assert out["run_result"].metadata["ok"] is False
    assert "NameError" in str(out["run_result"].value)


async def test_coder_run_without_upstream_raises():
    agent = CoderAgent(FakeLLM(), FakeOneShotSandbox(SandboxResult()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="generated_code"):
        await agent.execute(make_step("n2", type_="code_run"), {})


async def test_coder_framework_compare():
    llm = FakeLLM(responses=["print('langchain vs llamaindex')"])
    sandbox = FakeOneShotSandbox(SandboxResult(stdout="对比结果", exit_code=0))
    agent = CoderAgent(llm, sandbox)  # type: ignore[arg-type]

    out = await agent.execute(make_step("n1", type_="framework_compare"), {})
    assert "comparison_report" in out


async def test_coder_unsupported_type():
    agent = CoderAgent(FakeLLM(), FakeOneShotSandbox(SandboxResult()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="不支持"):
        await agent.execute(make_step("n1", type_="paper_parse"), {})


# ──────────────────────────────────────────────
# ResearchCodingAgent：确定性辅助函数
# ──────────────────────────────────────────────


def _make_agent(sandbox=None) -> ResearchCodingAgent:
    return ResearchCodingAgent(FakeLLM(), sandbox or FakeRepoSandbox([]))  # type: ignore[arg-type]


def test_validate_repo_url():
    agent = _make_agent()
    assert agent._validate_repo_url("https://github.com/user/repo") == "https://github.com/user/repo"
    with pytest.raises(ValueError, match="非法"):
        agent._validate_repo_url("https://evil.com/user/repo")


def test_extract_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "# comment\nnumpy==1.26.0\ntorch>=2.0\n-r extra.txt\ngit+https://github.com/x/y.git\n",
        encoding="utf-8",
    )
    agent = _make_agent()
    deps = agent._extract_requirements(str(tmp_path))
    assert deps == ["numpy", "torch"]  # 注释/-r/git+ 全部跳过


def test_detect_entrypoint_priority(tmp_path):
    (tmp_path / "main.py").write_text("")
    (tmp_path / "train.py").write_text("")
    agent = _make_agent()
    # main.py 在 train.py 之前
    assert agent._detect_entrypoint(str(tmp_path)) == "main.py"


def test_detect_entrypoint_fallback(tmp_path):
    (tmp_path / "zzz.py").write_text("")
    agent = _make_agent()
    assert agent._detect_entrypoint(str(tmp_path)) == "zzz.py"


# ──────────────────────────────────────────────
# ResearchCodingAgent：repo_discovery（monkeypatch GitHub 搜索）
# ──────────────────────────────────────────────


async def test_repo_discovery_picks_best(monkeypatch):
    candidates = [
        {"full_name": "a/low-star", "html_url": "https://github.com/a/low-star",
         "stars": 2, "pushed_at": NOW},
        {"full_name": "b/good", "html_url": "https://github.com/b/good",
         "stars": 1200, "pushed_at": NOW},
    ]
    monkeypatch.setattr(
        "scholar_agent.agent.workers.research_coding.search_repos",
        lambda query, **kw: candidates,
    )
    agent = _make_agent()

    step = make_step("n2", type_="repo_discovery",
                     agent="research_coding_agent",
                     inputs={"paper_title": "Attention Is All You Need"})
    out = await agent.execute(step, {})

    assert out["repo_url"].value == "https://github.com/b/good"
    assert out["repo_url"].type == ArtifactType.url


async def test_repo_discovery_no_qualified_raises(monkeypatch):
    monkeypatch.setattr(
        "scholar_agent.agent.workers.research_coding.search_repos",
        lambda query, **kw: [{"full_name": "a/dead", "html_url": "u",
                              "stars": 5, "pushed_at": "2020-01-01T00:00:00+00:00"}],
    )
    agent = _make_agent()
    with pytest.raises(ValueError, match="未找到合格"):
        await agent.execute(
            make_step("n2", type_="repo_discovery",
                      inputs={"paper_title": "x"}), {}
        )


# ──────────────────────────────────────────────
# ResearchCodingAgent：code_run 全流程（mock clone + 沙箱）
# ──────────────────────────────────────────────


def _prepare_workspace(tmp_path: Path) -> str:
    """伪造 clone 好的 repo：入口文件 + requirements。"""
    (tmp_path / "run.py").write_text("print('reproduced!')", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    return str(tmp_path)


async def test_repo_run_full_flow_success(tmp_path, monkeypatch):
    """clone → 装依赖（ok）→ 断网切换 → Harness 一次成功 → 清理。"""
    ws = _prepare_workspace(tmp_path)
    agent = _make_agent(FakeRepoSandbox([
        SandboxResult(stdout="pip done", exit_code=0),   # pip install
        SandboxResult(stdout="reproduced!", exit_code=0),  # python run.py
    ]))
    monkeypatch.setattr(agent, "_clone_and_setup", lambda url: _async_return(ws))
    step = make_step("n3", type_="code_run", agent="research_coding_agent",
                     required=["repo_url"])

    out = await agent.execute(step, {"repo_url": art("repo_url", "https://github.com/b/good", ArtifactType.url)})

    assert out["run_result"].metadata["ok"] is True
    assert out["run_result"].value == "reproduced!"
    assert agent.sandbox.cleaned == ["run-c1"]      # finally 清理了运行容器
    assert agent.sandbox.registry.removed == "n3"   # 注册表键 = step.id，已删


async def _async_return(value):
    return value


async def test_repo_run_install_failure(tmp_path, monkeypatch):
    """依赖装不上（LLM abort）→ run_result 记录失败，不进 Harness。"""
    ws = _prepare_workspace(tmp_path)
    sandbox = FakeRepoSandbox([
        SandboxResult(stderr="No matching distribution for numpy-x", exit_code=1),
    ])
    llm = FakeLLM(structured={"RepairAction": RepairAction(action="abort", reason="救不了")})
    agent = ResearchCodingAgent(llm, sandbox)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_clone_and_setup", lambda url: _async_return(ws))
    step = make_step("n3", type_="code_run")

    out = await agent.execute(step, {"repo_url": art("repo_url", "https://github.com/b/good", ArtifactType.url)})

    assert out["run_result"].metadata["ok"] is False
    assert out["run_result"].metadata["stage"] == "install"
    assert "依赖安装失败" in str(out["run_result"].value)


async def test_repo_run_harness_heals_failure(tmp_path, monkeypatch):
    """运行失败 → Harness 修复 → 第 2 次成功（stderr 不同防熔断）。"""
    ws = _prepare_workspace(tmp_path)
    agent = _make_agent(FakeRepoSandbox([
        SandboxResult(stdout="pip done", exit_code=0),
        SandboxResult(stderr="ImportError: no module utils", exit_code=1),
        SandboxResult(stdout="healed output", exit_code=0),
    ]))
    # Harness._repair_code 调 ainvoke：预设修复后的代码
    agent.llm = FakeLLM(responses=["print('reproduced!')  # fixed"])  # type: ignore[assignment]
    monkeypatch.setattr(agent, "_clone_and_setup", lambda url: _async_return(ws))
    step = make_step("n3", type_="code_run")

    out = await agent.execute(step, {"repo_url": art("repo_url", "https://github.com/b/good", ArtifactType.url)})

    assert out["run_result"].metadata["ok"] is True
    assert out["run_result"].value == "healed output"
    assert out["run_result"].metadata["attempts"] == 1  # 一次失败记录


async def test_repo_run_invalid_url_rejected():
    agent = _make_agent()
    with pytest.raises(ValueError, match="非法"):
        await agent.execute(
            make_step("n3", type_="code_run"),
            {"repo_url": art("repo_url", "https://evil.com/x", ArtifactType.url)},
        )
