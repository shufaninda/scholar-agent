"""测试 tools：GitHub 质量过滤 + 依赖两层恢复。"""

from types import SimpleNamespace

from tests.conftest import FakeLLM
from scholar_agent.agent.tools.dependency_installer import (
    PACKAGE_FIXES,
    RepairAction,
    _apply_action,
    install_dependencies,
)
from scholar_agent.agent.tools.github_search import pick_best_repo


# ─── GitHub 质量过滤 ───

def _recent_date() -> str:
    from datetime import datetime, timezone
    return str(datetime.now(timezone.utc))


def test_pick_best_repo_prefers_qualified():
    """star 达标 + 近期活跃的候选被选中。"""
    candidates = [
        {"full_name": "a/low", "html_url": "u1", "stars": 3, "pushed_at": _recent_date()},
        {"full_name": "b/good", "html_url": "u2", "stars": 500, "pushed_at": _recent_date()},
    ]
    assert pick_best_repo(candidates) == "u2"


def test_pick_best_repo_all_unqualified_returns_none():
    """全部低星 → None（调用方降级）。"""
    candidates = [
        {"full_name": "a/toy", "html_url": "u1", "stars": 1, "pushed_at": _recent_date()},
    ]
    assert pick_best_repo(candidates) is None


# ─── 依赖两层恢复 ───

class ScriptedSandbox:
    """脚本化沙箱：按顺序返回预设执行结果。"""

    def __init__(self, results):
        self.results = list(results)
        self.commands: list = []

    async def exec_in(self, sandbox_id, command):
        self.commands.append(command)
        return self.results.pop(0)


async def test_install_success_first_try():
    sb = ScriptedSandbox([SimpleNamespace(ok=True, stdout="ok", stderr="", exit_code=0, error="")])
    result = await install_dependencies(["numpy"], sb, "sid")
    assert result.success
    assert result.repaired is False
    assert len(sb.commands) == 1


async def test_rule_fix_recovers_bad_package():
    """第 1 层：opencv 拼错 → 规则表纠正后装成功，不花 LLM。"""
    sb = ScriptedSandbox([
        SimpleNamespace(ok=False, stdout="", stderr="no opencv", exit_code=1, error=""),
        SimpleNamespace(ok=True, stdout="", stderr="", exit_code=0, error=""),
    ])
    result = await install_dependencies(["opencv"], sb, "sid")
    assert result.success
    assert result.repaired is True
    # 第二次安装的命令里是纠正后的包名
    assert "opencv-python-headless" in sb.commands[1]


async def test_llm_fix_recovers_when_rule_fails():
    """第 2 层：规则救不了 → LLM 出 replace_package 动作。"""
    llm = FakeLLM(structured={"RepairAction": RepairAction(
        action="replace_package", replace_package="torc", with_package="torch",
    )})
    sb = ScriptedSandbox([
        SimpleNamespace(ok=False, stdout="", stderr="torc not found", exit_code=1, error=""),
        SimpleNamespace(ok=True, stdout="", stderr="", exit_code=0, error=""),
    ])
    result = await install_dependencies(["torc"], sb, "sid", llm=llm)
    assert result.success
    assert result.repaired is True
    assert sb.commands[1][-1] == "torch"


def test_apply_action_remove_package():
    deps = ["shutil", "numpy"]
    action = RepairAction(action="remove_package", remove_package="shutil")
    assert _apply_action(deps, action) == ["numpy"]


def test_apply_action_abort_keeps_deps():
    deps = ["torch"]
    action = RepairAction(action="abort")
    assert _apply_action(deps, action) == ["torch"]


def test_package_fixes_table_sane():
    assert PACKAGE_FIXES["cv2"] == "opencv-python-headless"
