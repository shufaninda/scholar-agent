"""测试 sandbox：安全校验（纯函数）+ 注册表（fakeredis）+ 执行器（mock Docker）。"""

import pytest

from scholar_agent.agent.sandbox.registry import SandboxRegistry
from scholar_agent.agent.sandbox.security import (
    authorize_mount_path,
    secure_docker_kwargs,
    validate_image,
)

# ─── 安全校验 ───

def test_validate_image_rejects_unknown():
    with pytest.raises(ValueError, match="白名单"):
        validate_image("ubuntu:latest")


def test_validate_image_accepts_allowlisted():
    assert validate_image("python:3.12-slim") == "python:3.12-slim"


def test_authorize_mount_path_allows_workspace_subdir(tmp_path):
    ok_path = str(tmp_path / "ws_repo")
    assert authorize_mount_path(ok_path) == str((tmp_path / "ws_repo").resolve())


def test_authorize_mount_path_blocks_escape(tmp_path, monkeypatch):
    """路径逃逸：/etc、../../etc 必须被拒。"""
    import scholar_agent.agent.sandbox.security as sec
    monkeypatch.setattr(sec, "WORKSPACE_ROOTS", [str(tmp_path)])
    with pytest.raises(ValueError, match="逃逸"):
        authorize_mount_path("/etc")
    with pytest.raises(ValueError, match="逃逸"):
        authorize_mount_path(str(tmp_path / ".." / ".." / "etc"))


def test_secure_docker_kwargs_merges_extra():
    kwargs = secure_docker_kwargs({"mem_limit": "1g", "network_mode": "bridge"})
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["mem_limit"] == "1g"       # extra 覆盖默认
    assert kwargs["network_mode"] == "bridge"


# ─── 注册表 ───

async def test_registry_roundtrip(fake_redis):
    reg = SandboxRegistry(fake_redis)
    await reg.register("plan1", "abc123")
    assert await reg.get("plan1") == "abc123"
    await reg.remove("plan1")
    assert await reg.get("plan1") is None


async def test_registry_get_missing_returns_none(fake_redis):
    reg = SandboxRegistry(fake_redis)
    assert await reg.get("ghost") is None


# ─── 执行器（mock Docker client）───

class FakeContainer:
    """mock 容器：exec_run 返回预设输出。"""

    def __init__(self, exec_output=None, not_found=False):
        self.id = "fake_container_id_000"
        self.exec_output = exec_output or (0, (b"ok", b""))
        self.not_found = not_found

    def exec_run(self, command, demux=False, workdir=None, **kwargs):
        if self.not_found:
            import docker.errors as docker_err
            raise docker_err.NotFound("container gone")
        return self.exec_output


class FakeContainersAPI:
    def __init__(self, container):
        self.container = container

    def get(self, container_id):
        return self.container

    def run(self, *args, **kwargs):
        return self.container


class FakeDockerClient:
    def __init__(self, container):
        self.containers = FakeContainersAPI(container)


async def test_exec_in_success(fake_redis):
    from scholar_agent.agent.sandbox.docker_executor import DockerSandbox

    container = FakeContainer(exec_output=(0, (b"hello", b"")))
    sandbox = DockerSandbox(SandboxRegistry(fake_redis), client=FakeDockerClient(container))
    result = await sandbox.exec_in("cid", ["python", "run.py"])
    assert result.ok
    assert result.stdout == "hello"


async def test_exec_in_nonzero_exit(fake_redis):
    from scholar_agent.agent.sandbox.docker_executor import DockerSandbox

    container = FakeContainer(exec_output=(1, (b"", b"Traceback...")))
    sandbox = DockerSandbox(SandboxRegistry(fake_redis), client=FakeDockerClient(container))
    result = await sandbox.exec_in("cid", ["python", "run.py"])
    assert not result.ok
    assert "Traceback" in result.stderr


async def test_exec_in_container_missing(fake_redis):
    """容器不存在：返回沙箱层错误（exit_code=-1），不抛异常。"""
    from scholar_agent.agent.sandbox.docker_executor import DockerSandbox

    container = FakeContainer(not_found=True)
    sandbox = DockerSandbox(SandboxRegistry(fake_redis), client=FakeDockerClient(container))
    result = await sandbox.exec_in("cid", ["python", "run.py"])
    assert result.exit_code == -1
    assert result.error
