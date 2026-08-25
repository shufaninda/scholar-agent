"""依赖安装工具：pip install + 两层恢复（规则兜底 + LLM 修复）。

被 CoderAgent 和 ResearchCodingAgent 调用。

两层恢复：
    第 1 层（规则，免费）：包名纠正表——opencv→opencv-python-headless 这类
    常见错误直接查表，不花 LLM token
    第 2 层（LLM，付费）：规则救不了再调 LLM 出修复动作（受控 JSON，不是任意 shell）
"""

import json
import logging

from pydantic import BaseModel

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.dependency_prompts import (
    DEPENDENCY_RECOVERY_SYSTEM,
    dependency_recovery_user_prompt,
)
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.agent.sandbox.result import SandboxResult

logger = logging.getLogger(__name__)

# 国内镜像源，容器里 pip 提速
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

MAX_INSTALL_ATTEMPTS = 3

# 规则兜底表：常见包名错误 → 正确包名（第 1 层恢复，零成本）
PACKAGE_FIXES: dict[str, str] = {
    "opencv": "opencv-python-headless",
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "pillow-simd": "pillow",
    "sklearn": "scikit-learn",
    "tensorflow-gpu": "tensorflow",
}


class InstallResult(BaseModel):
    """依赖安装结果。"""

    success: bool
    repaired: bool = False   # 是否经过修复（规则或 LLM）
    reason: str = ""


class RepairAction(BaseModel):
    """LLM 修复动作的受控 schema（只能出有限动作，不是任意 shell）。"""

    action: str  # remove_package / replace_package / upgrade_python / rewrite_dependencies / abort
    reason: str = ""
    remove_package: str = ""
    replace_package: str = ""
    with_package: str = ""
    target_image: str = ""
    next_dependencies: list[str] = []


async def install_dependencies(
    deps: list[str],
    sandbox: DockerSandbox,
    sandbox_id: str,
    llm: LLMClient | None = None,
) -> InstallResult:
    """装依赖 + 失败两层恢复。最多 MAX_INSTALL_ATTEMPTS 轮。"""
    current = list(deps)
    repaired = False

    for _ in range(MAX_INSTALL_ATTEMPTS):
        result = await _pip_install(sandbox, sandbox_id, current)
        if result.ok:
            return InstallResult(success=True, repaired=repaired)

        # 第 1 层：规则兜底（零成本先试）
        fixed, current = _rule_based_fix(current, result.stderr)
        if fixed:
            repaired = True
            logger.info("dep_fix_by_rule new_deps=%s", current)
            continue

        # 第 2 层：LLM 修复（规则救不了）
        if llm is None:
            return InstallResult(success=False, repaired=repaired, reason=result.stderr[:500])
        action, current, changed = await _llm_repair(llm, current, result.stderr)
        if not changed or action == "abort":
            return InstallResult(success=False, repaired=repaired, reason=result.stderr[:500])
        repaired = True

    return InstallResult(success=False, repaired=repaired, reason="依赖安装重试耗尽")


async def _pip_install(sandbox: DockerSandbox, sandbox_id: str, deps: list[str]) -> SandboxResult:
    """在持久化容器里执行 pip install。"""
    if not deps:
        return SandboxResult(stdout="no deps", exit_code=0)
    cmd = ["pip", "install", "-i", PIP_INDEX, *deps]
    return await sandbox.exec_in(sandbox_id, cmd)


def _rule_based_fix(deps: list[str], _error: str) -> tuple[bool, list[str]]:
    """第 1 层恢复：查表纠正包名。有改动返回 (True, 新列表)。"""
    fixed = [PACKAGE_FIXES.get(d, d) for d in deps]
    return (fixed != deps), fixed


async def _llm_repair(
    llm: LLMClient, deps: list[str], pip_error: str
) -> tuple[str, list[str], bool]:
    """第 2 层恢复：LLM 出受控修复动作。返回 (action, 新列表, 是否有改动)。"""
    structured = llm.with_structured_output(RepairAction)
    action = await structured.ainvoke([
        {"role": "system", "content": DEPENDENCY_RECOVERY_SYSTEM},
        {"role": "user", "content": dependency_recovery_user_prompt(
            json.dumps(deps), pip_error[-2000:]
        )},
    ])
    new_deps = _apply_action(deps, action)
    return action.action, new_deps, (new_deps != deps or action.action == "upgrade_python")


def _apply_action(deps: list[str], action: RepairAction) -> list[str]:
    """把受控动作应用到依赖列表（每种 action 一个分支，禁止任意 shell）。"""
    if action.action == "remove_package" and action.remove_package in deps:
        return [d for d in deps if d != action.remove_package]
    if action.action == "replace_package" and action.replace_package in deps:
        return [action.with_package if d == action.replace_package else d for d in deps]
    if action.action == "rewrite_dependencies" and action.next_dependencies:
        return list(action.next_dependencies)
    return deps  # upgrade_python 不改列表（换镜像），abort 不动
