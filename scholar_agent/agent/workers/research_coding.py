"""ResearchCodingAgent：论文仓库发现 + 执行 + Harness 自愈。

⭐ 关键设计：repo_discovery / repo_prepare 绕过 LLM 走确定性
后端逻辑——LLM 会编造 URL 或选错文件，破坏可重复性。只有 Harness 修复
代码阶段才调 LLM。

步骤契约（对齐 planner/templates.py）：
    repo_discovery: 输入 paper_title → 输出 repo_url artifact（不调 LLM）
    code_run:       消费 repo_url → clone → 装依赖 → Harness 自愈执行
                    → 输出 run_result artifact
"""

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from scholar_agent.agent.harness.runner import Harness
from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.agent.tools.dependency_installer import install_dependencies
from scholar_agent.agent.tools.github_search import pick_best_repo, search_repos
from scholar_agent.agent.workers.base import BaseAgent
from scholar_agent.models.artifact import Artifact, ArtifactType
from scholar_agent.models.step import Step

logger = logging.getLogger(__name__)

# 入口文件探测顺序（确定性，不调 LLM）
ENTRYPOINT_CANDIDATES = [
    "run.py", "main.py", "train.py", "test.py", "app.py",
    "src/run.py", "src/main.py", "src/train.py",
]

GITHUB_URL_PATTERN = r"^https://github\.com/[^/]+/[^/]+(?:\.git)?$"


class ResearchCodingAgent(BaseAgent):
    """论文仓库执行 Agent：搜 repo → clone → 装依赖 → Harness 自愈运行。

    关键：repo_discovery / clone / 入口探测 全部确定性，不调 LLM。
    只有 Harness 修复代码阶段才调 LLM。
    """

    name = "research_coding_agent"

    def __init__(self, llm: LLMClient, sandbox: DockerSandbox):
        super().__init__(llm=llm, sandbox=sandbox)

    async def execute(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """按步骤类型分发：repo_discovery / code_run。"""
        if step.type == "repo_discovery":
            return await self._discover_repo(step, artifacts)
        if step.type == "code_run":
            return await self._run_repo(step, artifacts)
        raise ValueError(f"ResearchCodingAgent 不支持的步骤类型: {step.type}")

    # ──────────────────────────────────────────
    # 阶段1：repo_discovery（确定性，不调 LLM）
    # ──────────────────────────────────────────

    async def _discover_repo(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """搜 GitHub repo：PyGithub API + 质量过滤（star/活跃度）。"""
        # 搜索词优先级：inputs 指定 > 上游 parsed_paper > 步骤描述
        paper_title = step.inputs.get("paper_title") or ""
        if not paper_title and "parsed_paper" in artifacts:
            paper_title = str(artifacts["parsed_paper"].value)[:200]
        if not paper_title:
            paper_title = step.description or step.name

        query = f"{paper_title} implementation"
        candidates = await asyncio.to_thread(search_repos, query)

        repo_url = pick_best_repo(candidates)
        if repo_url is None:
            raise ValueError(
                f"未找到合格的复现仓库（star≥10 且近一年活跃）: {paper_title}"
            )
        logger.info("repo_discovered url=%s", repo_url)

        return {
            "repo_url": Artifact(
                key="repo_url",
                type=ArtifactType.url,
                value=repo_url,
                producer_task_id=step.id,
                metadata={"candidates": len(candidates)},
            )
        }

    # ──────────────────────────────────────────
    # 阶段2-4：clone → 装依赖 → Harness 自愈执行
    # ──────────────────────────────────────────

    async def _run_repo(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """执行 repo：clone → 持久化沙箱装依赖 → 断网 → Harness 跑入口。"""
        if "repo_url" not in artifacts:
            raise ValueError("code_run 缺少上游产物 repo_url")
        repo_url = self._validate_repo_url(str(artifacts["repo_url"].value))
        plan_id = step.id  # 沙箱 registry 键（只需唯一性，步骤 ID 足够）

        workspace = await self._clone_and_setup(repo_url)
        await self._verify_repo_revision(workspace, step.inputs.get("expected_commit"))

        run_container: str | None = None
        try:
            # 装依赖阶段：bridge 联网容器
            install_container = await self.sandbox.create_for_install(
                workspace, plan_id
            )
            deps = self._extract_requirements(workspace)
            install = await install_dependencies(
                deps, self.sandbox, install_container, llm=self.llm
            )
            if not install.success:
                return {
                    "run_result": Artifact(
                        key="run_result",
                        type=ArtifactType.text,
                        value=f"依赖安装失败: {install.reason}",
                        producer_task_id=step.id,
                        metadata={"ok": False, "stage": "install"},
                    )
                }

            # 两阶段网络：装完依赖切断网容器（workspace 卷保留依赖）
            run_container = await self.sandbox.switch_to_run_phase(
                install_container
            )

            entrypoint = self._detect_entrypoint(workspace)
            harness = Harness(self.llm, self.sandbox)
            report = await harness.run_with_healing(
                workspace=workspace,
                sandbox_id=run_container,   # ⭐ 复用持久化沙箱
                entrypoint=entrypoint,
            )

            output = (
                report.final_result
                if report.status == "passed"
                else f"运行失败（{report.reason}）\n尝试记录: "
                     + "; ".join(f"#{a.attempt} exit={a.exit_code}" for a in report.attempts)
            )
            return {
                "run_result": Artifact(
                    key="run_result",
                    type=ArtifactType.text,
                    value=output,
                    producer_task_id=step.id,
                    metadata={
                        "ok": report.status == "passed",
                        "repo_url": repo_url,
                        "attempts": len(report.attempts),
                    },
                )
            }
        finally:
            if run_container:
                await self.sandbox.cleanup(run_container)
                await self.sandbox.registry.remove(plan_id)

    # ──────────────────────────────────────────
    # 确定性辅助（全部不调 LLM）
    # ──────────────────────────────────────────

    def _validate_repo_url(self, url: str) -> str:
        """校验 repo URL 格式，防 LLM 介入编造 URL。"""
        if not re.match(GITHUB_URL_PATTERN, url):
            raise ValueError(f"非法 repo URL: {url}")
        return url

    async def _clone_and_setup(self, repo_url: str) -> str:
        """确定性 clone：多策略降级（shallow → full → codeload tar.gz）。

        subprocess 是同步的，用 to_thread 包住避免阻塞事件循环。
        """
        workspace = tempfile.mkdtemp(prefix="scholar_")

        def _shallow() -> None:
            subprocess.run(  # noqa: S603 S607
                ["git", "clone", "--depth", "1", repo_url, workspace],
                check=True, timeout=60, capture_output=True,
            )

        def _full() -> None:
            subprocess.run(  # noqa: S603 S607
                ["git", "clone", repo_url, workspace],
                check=True, timeout=180, capture_output=True,
            )

        def _tarball() -> None:
            match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
            if not match:
                raise RuntimeError(f"无法解析 repo URL: {repo_url}")
            owner, name = match.groups()
            tar_url = f"https://codeload.github.com/{owner}/{name}/tar.gz/refs/heads/main"
            tar_path = f"{workspace}.tar.gz"
            subprocess.run(["curl", "-sL", "-o", tar_path, tar_url],  # noqa: S603 S607
                           check=True, timeout=120, capture_output=True)
            subprocess.run(["tar", "xzf", tar_path, "-C", workspace,  # noqa: S603 S607
                            "--strip-components=1"], check=True, capture_output=True)
            Path(tar_path).unlink(missing_ok=True)

        for label, strategy in (("shallow", _shallow), ("full", _full), ("tarball", _tarball)):
            try:
                await asyncio.to_thread(strategy)
                logger.info("repo_cloned url=%s strategy=%s", repo_url, label)
                return workspace
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
                logger.warning("clone_strategy_failed url=%s strategy=%s", repo_url, label)
                shutil.rmtree(workspace, ignore_errors=True)
                Path(workspace).mkdir(parents=True, exist_ok=True)  # tarball 复用目录

        shutil.rmtree(workspace, ignore_errors=True)
        raise RuntimeError(f"所有 clone 策略均失败: {repo_url}")

    async def _verify_repo_revision(self, workspace: str, expected_commit: str | None) -> None:
        """校验 commit hash：复现可重复性的根本。未指定则跳过。

        subprocess 同步阻塞，to_thread 包住防卡事件循环（同 _clone_and_setup）。
        """
        if not expected_commit:
            return
        if not re.match(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$", expected_commit):
            raise ValueError(f"expected_commit 格式不合法: {expected_commit}")

        def _rev_parse() -> str:
            return subprocess.run(  # noqa: S603 S607
                ["git", "rev-parse", "HEAD"], cwd=workspace,
                capture_output=True, text=True, check=True,
            ).stdout.strip()

        actual = await asyncio.to_thread(_rev_parse)
        if not actual.startswith(expected_commit):
            raise RuntimeError(
                f"commit 不匹配: expected {expected_commit}, got {actual}"
            )

    def _extract_requirements(self, workspace: str) -> list[str]:
        """从 requirements.txt 提取依赖列表（确定性解析，不调 LLM）。"""
        req = Path(workspace) / "requirements.txt"
        if not req.exists():
            return []
        deps = []
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-", "git+")):
                continue  # 跳过注释/选项/git 依赖（沙箱装不了）
            deps.append(re.split(r"[=<>~\[]", line, maxsplit=1)[0])
        return deps

    def _detect_entrypoint(self, workspace: str) -> str:
        """探测入口文件：按候选顺序找第一个存在的（确定性，不调 LLM）。"""
        root = Path(workspace)
        for candidate in ENTRYPOINT_CANDIDATES:
            if (root / candidate).is_file():
                return candidate
        py_files = sorted(root.glob("*.py"))
        return py_files[0].name if py_files else "run.py"
