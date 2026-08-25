"""Docker 沙箱执行器：四层隔离 + 两阶段网络 + 持久化容器复用。

四层隔离：
    1. 文件系统：只挂载授权过的 workspace（authorize_mount_path）
    2. 网络：运行阶段 network_mode="none" 断网
    3. 资源：mem_limit + cpu_quota + pids_limit
    4. 权限：user=nobody + cap_drop=ALL + no-new-privileges

两阶段网络（解决"装依赖要网 / 跑代码不许网"矛盾）：
    install 阶段 bridge（能 pip install）→ run 阶段 none（断网防外联）

docker-py 是同步 SDK，统一用 asyncio.to_thread 包住避免阻塞事件循环。
"""

import asyncio
import logging
import time

import docker
from docker.errors import APIError, NotFound

from scholar_agent.agent.sandbox.registry import SandboxRegistry
from scholar_agent.agent.sandbox.result import SandboxResult
from scholar_agent.agent.sandbox.security import (
    authorize_mount_path,
    secure_docker_kwargs,
    validate_image,
)

logger = logging.getLogger(__name__)

# 所有沙箱容器打这个 label，启动钩子按它清理孤儿容器
SANDBOX_LABEL = "scholar-agent"


class DockerSandbox:
    """Docker 沙箱：一次性执行 + 持久化容器两种模式。

    持久化模式生命周期：create_for_install（联网）→ 装依赖
    → switch_to_run_phase（断网）→ exec_in 反复执行 → cleanup。
    """

    def __init__(self, registry: SandboxRegistry, client: docker.DockerClient | None = None):
        self.registry = registry
        # 惰性连接：单测注入 mock client，不真连 Docker
        self._client = client

    @property
    def client(self) -> docker.DockerClient:
        """惰性初始化 Docker 客户端（导入时不连，测试可注入）。"""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    # ──────────────────────────────────────────
    # 一次性执行（简单场景：跑一段代码拿结果）
    # ──────────────────────────────────────────

    async def execute(
        self,
        code: str,
        image: str = "python:3.12-slim",
        timeout: int = 300,
    ) -> SandboxResult:
        """一次性执行：起容器 → 跑 python -c → wait → 清理。"""
        validate_image(image)
        start = time.monotonic()
        try:
            container = await asyncio.to_thread(
                self.client.containers.run,
                image,
                ["python", "-c", code],
                detach=True,
                network_mode="none",           # 断网
                mem_limit="512m",
                cpu_quota=100_000,             # 1 核
                labels={SANDBOX_LABEL: "true"},
                **secure_docker_kwargs(),
            )
            result = await self._wait_and_collect(container, timeout)
            await asyncio.to_thread(container.remove, force=True)
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except APIError as exc:
            return SandboxResult(exit_code=-1, error=f"docker api error: {exc}")

    async def _wait_and_collect(self, container, timeout: int) -> SandboxResult:
        """等容器退出并收集 stdout/stderr（同步 wait 用 to_thread 包）。"""
        try:
            await asyncio.to_thread(container.wait, timeout=timeout)
        except Exception as exc:  # 超时/连接异常：尽力收集已有输出
            logger.warning("sandbox_wait_failed: %s", exc)
        logs = await asyncio.to_thread(container.logs, stdout=True, stderr=True)
        exit_code = await asyncio.to_thread(
            lambda: container.attrs.get("State", {}).get("ExitCode", -1)
        )
        text = logs.decode(errors="replace") if isinstance(logs, bytes) else str(logs)
        return SandboxResult(stdout=text, exit_code=exit_code)

    # ──────────────────────────────────────────
    # 持久化容器（复用模式：clone + 装依赖后反复执行）
    # ──────────────────────────────────────────

    async def create_for_install(
        self, workspace: str, plan_id: str, image: str = "python:3.12-slim"
    ) -> str:
        """装依赖阶段容器：bridge 联网（pip 要下载），资源放宽到 1g。"""
        return await self._create_persistent(
            workspace, plan_id, image, network_mode="bridge",
            mem_limit="1g", cpu_quota=200_000, phase="install",
        )

    async def create_persistent(
        self, workspace: str, plan_id: str, image: str = "python:3.12-slim"
    ) -> str:
        """运行阶段容器：直接断网创建（依赖已装好的场景）。"""
        return await self._create_persistent(
            workspace, plan_id, image, network_mode="none",
            mem_limit="512m", cpu_quota=100_000, phase="run",
        )

    async def _create_persistent(
        self, workspace: str, plan_id: str, image: str,
        network_mode: str, mem_limit: str, cpu_quota: int, phase: str,
    ) -> str:
        """持久化容器创建的公共路径：校验 → 起容器 → 注册 Redis。"""
        validate_image(image)
        host_ws = authorize_mount_path(workspace)  # 路径逃逸校验
        container = await asyncio.to_thread(
            self.client.containers.run,
            image,
            command=["sleep", "3600"],           # 常驻等 exec
            detach=True,
            tty=True,
            volumes={host_ws: {"bind": "/workspace", "mode": "rw"}},
            network_mode=network_mode,
            working_dir="/workspace",
            labels={SANDBOX_LABEL: "true", "plan_id": plan_id, "phase": phase},
            **secure_docker_kwargs({"mem_limit": mem_limit, "cpu_quota": cpu_quota}),
        )
        await self.registry.register(plan_id, container.id)
        logger.info("sandbox_created plan=%s phase=%s id=%s", plan_id, phase, container.id[:12])
        return container.id

    async def switch_to_run_phase(self, container_id: str) -> str:
        """切换到运行阶段：停删旧容器 → 断网重建（workspace 卷保留依赖）。

        依赖装在 /workspace/.venv（挂载卷里），重建容器依赖不丢。
        """
        old = self.client.containers.get(container_id)
        plan_id = old.labels.get("plan_id", "")
        workspace_label = old.labels.get("phase", "")
        attrs_volumes = old.attrs.get("Mounts", [])
        await asyncio.to_thread(old.remove, force=True)

        # 从旧容器的 Mounts 里找回 workspace 宿主机路径
        host_ws = next(
            (m["Source"] for m in attrs_volumes if m.get("Destination") == "/workspace"),
            None,
        )
        if host_ws is None:  # 找不到挂载信息，无法安全重建
            raise APIError("原容器无 /workspace 挂载，无法切换运行阶段")
        _ = workspace_label
        return await self.create_persistent(host_ws, plan_id)

    async def exec_in(self, container_id: str, command: list[str]) -> SandboxResult:
        """在持久化容器里执行命令（demux 分离 stdout/stderr）。"""
        start = time.monotonic()
        try:
            container = self.client.containers.get(container_id)
        except NotFound:
            return SandboxResult(exit_code=-1, error=f"容器 {container_id[:12]} 不存在")

        try:
            # demux=True 返回 (exit_code, (stdout_bytes, stderr_bytes))
            exit_code, (out, err) = await asyncio.to_thread(
                container.exec_run, command, demux=True, workdir="/workspace"
            )
            return SandboxResult(
                stdout=(out or b"").decode(errors="replace"),
                stderr=(err or b"").decode(errors="replace"),
                exit_code=exit_code,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except NotFound:
            return SandboxResult(exit_code=-1, error="容器执行期间被删除")
        except APIError as exc:
            return SandboxResult(exit_code=-1, error=f"docker api error: {exc}")

    async def cleanup(self, container_id: str) -> None:
        """清理容器：stop + force remove（幂等，不存在不报错）。"""
        try:
            container = self.client.containers.get(container_id)
            await asyncio.to_thread(container.stop, timeout=5)
            await asyncio.to_thread(container.remove, force=True)
            logger.info("sandbox_cleaned id=%s", container_id[:12])
        except NotFound:
            pass  # 幂等：已清理过就跳过
