"""机制2：SHA256 + Repository Fingerprint（防代码偷换）。

对应 DESIGN.md 2.4 节"机制 2"。
对应 Sea 的 AdapterCodeSHA256 + benchmarkRepositoryFingerprint。

为什么需要：
    1. 防"修复后代码没变"：LLM 可能返回原代码，靠 SHA256 检测
    2. 防"执行期间偷改源码"：LLM 可能在 run.py 里偷偷改其他文件
"""

import hashlib
from pathlib import Path

# 指纹采集范围：只有这些后缀的文件参与指纹（其余文件改动不影响判定）
FINGERPRINT_SUFFIXES = {".py", ".pyi", ".sh"}
# 白名单目录：这些目录下的变更自动忽略（依赖管理/调试产物/缓存）
IGNORE_DIRS = (".git", ".scholar", "__pycache__", ".venv", "node_modules")


def code_sha256(code: str) -> str:
    """单文件代码哈希，检测 LLM 修复前后是否真的变了。"""
    return hashlib.sha256(code.encode()).hexdigest()


def repo_fingerprint(workspace: str, max_file_size: int = 2_000_000) -> dict[str, str]:
    """仓库指纹：遍历所有 .py/.pyi/.sh，记录相对路径 → 内容 SHA256。

    用法：执行前 snapshot1 = repo_fingerprint()，执行后 snapshot2 = repo_fingerprint()。
    如果 snapshot1 != snapshot2 且改动文件不在白名单，判定为偷改。

    - 忽略 .git / .scholar / __pycache__ / .venv / node_modules
    - 忽略超过 max_file_size 的大文件（默认 2MB）
    """
    fingerprint: dict[str, str] = {}
    root = Path(workspace)
    if not root.exists():
        return fingerprint

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_str = path.relative_to(root).as_posix()
        if any(part in IGNORE_DIRS for part in Path(rel_str).parts):
            continue
        if path.suffix not in FINGERPRINT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_file_size:
                continue
            fingerprint[rel_str] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue  # 读取失败（权限/竞态删除）的文件跳过，不让指纹阶段崩溃
    return fingerprint


def detect_unauthorized_changes(
    before: dict[str, str], after: dict[str, str]
) -> list[str]:
    """检测执行期间的未授权文件变更。

    返回变更文件路径列表（相对 workspace 的 posix 路径）。
    新增/删除/内容变化都算变更。
    """
    changes = []
    for path in set(before) | set(after):
        if before.get(path) != after.get(path):
            changes.append(path)
    return sorted(changes)
