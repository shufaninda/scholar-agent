"""沙箱安全加固：镜像白名单 + 挂载路径授权 + cap-drop + no-new-privileges。

为什么需要：
    原 DockerSandbox 只设了 user="nobody" + network_mode="none"，缺少
    关键安全层。挂载路径不校验会路径逃逸，没 cap-drop 仍能调用危险系统调用，
    没镜像白名单可能拉恶意镜像。

关键修复（P1-7）：
    字符串 startswith 前缀匹配可被 /tmpevil 绕过，必须用 Path.parents 比较。
"""

import os
import tempfile
from pathlib import Path

# 镜像白名单：只允许这些基础镜像，防 LLM 在 task.inputs 里塞恶意镜像
IMAGE_ALLOWLIST = {
    "python:3.12-slim",
    "python:3.11-slim",
    "python:3.10-slim",
}

# 工作区根目录：所有挂载路径必须落在这下面，防路径逃逸
WORKSPACE_ROOTS = [os.environ.get("SANDBOX_WORKSPACE_ROOTS", tempfile.gettempdir())]


def validate_image(image: str) -> str:
    """镜像白名单校验。不在白名单的拒绝执行。"""
    if image not in IMAGE_ALLOWLIST:
        raise ValueError(f"镜像 {image} 不在白名单 {IMAGE_ALLOWLIST}")
    return image


def authorize_mount_path(path: str) -> str:
    """挂载路径授权：Abs + EvalSymlinks + 必须落在 WORKSPACE_ROOTS 下。

    防 LLM 通过 task.inputs["workspace"] 传入 "/etc" 或 "../../etc" 之类逃逸路径。

    ⚠️ 关键修复（P1-7）：不能用 str.startswith，会被 /tmpevil 绕过。
       必须用 Path.parents 比较：abs_path == root 或 root in abs_path.parents。
    """
    abs_path = Path(path).resolve()
    for root in WORKSPACE_ROOTS:
        root_abs = Path(root).resolve()
        # 精确父目录比较：root 本身或它的直接/间接子目录才放行
        if abs_path == root_abs or root_abs in abs_path.parents:
            return str(abs_path)
    raise ValueError(f"挂载路径 {path} 逃逸出授权工作区 {WORKSPACE_ROOTS}")


def secure_docker_kwargs(extra: dict | None = None) -> dict:
    """统一的 Docker 安全参数：cap-drop + no-new-privileges + pids-limit + tmpfs。"""
    secured = {
        "cap_drop": ["ALL"],                       # 丢掉全部 Linux capabilities
        "security_opt": ["no-new-privileges:true"],# 禁止 sudo/suid 提权
        "pids_limit": 256,                         # 防进程炸弹 fork 炸沙箱
        "read_only": False,
        "tmpfs": {"/tmp": "rw,noexec,nosuid,size=512m"},  # /tmp 可写但不可执行
        "user": "nobody",                          # 非 root 运行
    }
    secured.update(extra or {})
    return secured
