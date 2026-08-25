"""Harness 5 大机制 + runner 主循环的功能测试。

测试原则：FakeSandbox 脚本化 exec_in 返回值，FakeLLM 预设修复代码。
覆盖：一次成功 / 失败后修复成功 / Patch Policy 拦截 / 指标造假拦截 /
CircuitBreaker 熔断 / 代码未变化跳过 / 入口文件还原。
"""

import json
from pathlib import Path

from tests.conftest import FakeLLM

from scholar_agent.agent.harness.circuit_breaker import CircuitBreaker
from scholar_agent.agent.harness.feedback_injector import build_feedback_prompt
from scholar_agent.agent.harness.fingerprint import (
    code_sha256,
    detect_unauthorized_changes,
    repo_fingerprint,
)
from scholar_agent.agent.harness.metrics_recompute import (
    recompute_metrics,
    verify_metrics,
)
from scholar_agent.agent.harness.patch_policy import validate_patch
from scholar_agent.agent.harness.runner import Harness
from scholar_agent.agent.sandbox.result import SandboxResult
from scholar_agent.models.harness import HarnessAttempt


class FakeSandbox:
    """脚本化沙箱：按顺序弹出预设的 exec_in 结果。"""

    def __init__(self, results: list[SandboxResult]):
        self.results = list(results)
        self.calls: list[list[str]] = []

    async def exec_in(self, container_id: str, command: list[str]) -> SandboxResult:
        self.calls.append(command)
        if not self.results:
            return SandboxResult(stdout="", exit_code=0)
        return self.results.pop(0)


def make_workspace(tmp_path: Path, code: str) -> str:
    """构造带入口文件的 workspace。"""
    (tmp_path / "run.py").write_text(code, encoding="utf-8")
    return str(tmp_path)


# ──────────────────────────────────────────────
# 机制1：Patch Policy
# ──────────────────────────────────────────────


def test_patch_policy_blocks_pip_install():
    ok, reason = validate_patch("import os\nos.system('pip install torch')")
    assert not ok
    assert "装包" in reason


def test_patch_policy_blocks_fake_metric():
    ok, reason = validate_patch("mock_metric = 0.99")
    assert not ok
    assert "伪造指标" in reason


def test_patch_policy_allows_clean_code():
    ok, reason = validate_patch("print('hello world')")
    assert ok and reason == ""


# ──────────────────────────────────────────────
# 机制2：Fingerprint
# ──────────────────────────────────────────────


def test_code_sha256_changes_with_content():
    assert code_sha256("a") != code_sha256("b")
    assert code_sha256("a") == code_sha256("a")


def test_repo_fingerprint_ignores_git_and_pycache(tmp_path):
    (tmp_path / "run.py").write_text("print(1)")
    (tmp_path / "__pycache__" / "run.pyc").parent.mkdir(exist_ok=True)
    (tmp_path / "__pycache__" / "run.pbc").write_text("junk")
    (tmp_path / ".git" / "config").parent.mkdir(exist_ok=True)
    (tmp_path / ".git" / "config").write_text("junk")

    fp = repo_fingerprint(str(tmp_path))
    assert list(fp) == ["run.py"]  # 只采集 .py/.pyi/.sh，白名单目录被忽略


def test_detect_unauthorized_changes_all_kinds():
    before = {"a.py": "1", "b.py": "2", "c.py": "3"}
    after = {"a.py": "1", "b.py": "changed", "d.py": "new"}
    changes = detect_unauthorized_changes(before, after)
    assert set(changes) == {"b.py", "c.py", "d.py"}  # 改 + 删 + 增


# ──────────────────────────────────────────────
# 机制3：指标重算防伪
# ──────────────────────────────────────────────


def test_recompute_metrics_classification(tmp_path):
    preds = tmp_path / "predictions.jsonl"
    lines = [
        {"y_true": "cat", "y_pred": "cat"},
        {"y_true": "cat", "y_pred": "cat"},
        {"y_true": "dog", "y_pred": "dog"},
        {"y_true": "dog", "y_pred": "cat"},
    ]
    preds.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    metrics = recompute_metrics(str(preds))
    assert abs(metrics["accuracy"] - 0.75) < 1e-9


def test_recompute_metrics_regression(tmp_path):
    """回归任务：y_true 唯一值 ≥ 20 才按回归算（< 20 视为分类）。"""
    preds = tmp_path / "predictions.jsonl"
    # 25 个不同真值 → 唯一值 ≥ 20 → 回归
    lines = [
        {"y_true": float(i), "y_pred": float(i) + 0.5} for i in range(25)
    ]
    preds.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    metrics = recompute_metrics(str(preds))
    assert abs(metrics["mse"] - 0.25) < 1e-9  # 每个样本误差恒为 0.5
    assert abs(metrics["mae"] - 0.5) < 1e-9


def test_recompute_metrics_empty(tmp_path):
    preds = tmp_path / "predictions.jsonl"
    preds.write_text("", encoding="utf-8")
    assert recompute_metrics(str(preds)) == {"error": "empty predictions"}


def test_verify_metrics_detects_fraud():
    recomputed = {"accuracy": 0.75, "macro_f1": 0.7}
    ok, _ = verify_metrics({"accuracy": 0.75, "macro_f1": 0.7}, recomputed)
    assert ok

    ok, reason = verify_metrics({"accuracy": 0.99, "macro_f1": 0.7}, recomputed)
    assert not ok and "疑似伪造" in reason

    ok, reason = verify_metrics({"accuracy": 0.75}, recomputed)
    assert not ok and "缺少指标" in reason


# ──────────────────────────────────────────────
# 机制4：CircuitBreaker
# ──────────────────────────────────────────────


def test_circuit_breaker_breaks_on_identical_failures():
    breaker = CircuitBreaker(window=2)
    err = "Traceback ... error: boom"
    # 第 1 次失败：记录不熔断
    assert breaker.should_break("python run.py", err) == (False, "")
    # 第 2 次完全相同：熔断
    should, reason = breaker.should_break("python run.py", err)
    assert should and "死循环" in reason


def test_circuit_breaker_ignores_success_and_different_errors():
    breaker = CircuitBreaker(window=2)
    # 成功输出不记录
    assert breaker.should_break("cmd", "all good") == (False, "")
    # 两次不同失败：不熔断
    assert breaker.should_break("cmd", "error: one") == (False, "")
    assert breaker.should_break("cmd", "error: two") == (False, "")


# ──────────────────────────────────────────────
# 机制5：FeedbackInjector
# ──────────────────────────────────────────────


def test_feedback_prompt_empty_and_filled():
    assert build_feedback_prompt([]) == ""
    prompt = build_feedback_prompt([
        HarnessAttempt(attempt=1, exit_code=1, error="NameError: name 'x' is not defined"),
    ])
    assert "DO NOT try these approaches again" in prompt
    assert "Attempt 1 (exit_code=1)" in prompt


# ──────────────────────────────────────────────
# runner：run_with_healing 主循环
# ──────────────────────────────────────────────


async def test_harness_passes_first_attempt(tmp_path):
    """一次成功：直接 passed，入口文件内容被还原。"""
    original = "print('ok')"
    ws = make_workspace(tmp_path, original)
    sandbox = FakeSandbox([SandboxResult(stdout="ok", exit_code=0)])
    harness = Harness(FakeLLM(), sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py")

    assert report.status == "passed"
    assert report.final_result == "ok"
    assert report.attempts == []  # 一次成功没有失败记录
    assert Path(ws, "run.py").read_text(encoding="utf-8") == original  # 已还原


async def test_harness_repairs_and_passes_second_attempt(tmp_path):
    """第 1 次失败 → LLM 修复 → 第 2 次成功。"""
    ws = make_workspace(tmp_path, "print('bad')")
    # 两次失败的 stderr 必须不同，否则 CircuitBreaker(window=2) 会熔断
    sandbox = FakeSandbox([
        SandboxResult(stderr="NameError: name 'x' is not defined", exit_code=1),
        SandboxResult(stdout="fixed", exit_code=0),
    ])
    llm = FakeLLM(responses=["print('fixed')"])
    harness = Harness(llm, sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py")

    assert report.status == "passed"
    assert len(report.attempts) == 1          # 只有第 1 次的失败记录
    assert report.attempts[0].repaired is False
    assert len(llm.calls) == 1                # 调了一次 LLM 修复
    assert "DO NOT try" in llm.calls[0][0]["content"]  # 机制5 负反馈已注入 system


async def test_harness_breaks_on_identical_failures(tmp_path):
    """连续两次相同失败 → CircuitBreaker 熔断，提前返回失败。"""
    ws = make_workspace(tmp_path, "print('bad')")
    same_err = "error: same failure"
    sandbox = FakeSandbox([
        SandboxResult(stderr=same_err, exit_code=1),
        SandboxResult(stderr=same_err, exit_code=1),
    ])
    # LLM 每次返回不同代码（绕过"代码未变化"检查），让熔断逻辑主导
    llm = FakeLLM(responses=["print('fix1')", "print('fix2')"])
    harness = Harness(llm, sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py", max_attempts=3)

    assert report.status == "failed"
    assert "死循环" in report.reason
    assert len(report.attempts) == 2  # 第 3 次没跑（熔断省了一次）


async def test_harness_skips_unchanged_repair(tmp_path):
    """LLM 返回原代码（没变）→ 第 2 次跳过执行。"""
    ws = make_workspace(tmp_path, "print('bad')")
    sandbox = FakeSandbox([
        SandboxResult(stderr="error: first", exit_code=1),
        # 如果第 2 次真的执行了会消费这个成功结果——测试要求它不被消费
        SandboxResult(stdout="should not run", exit_code=0),
    ])
    llm = FakeLLM(responses=["print('bad')"])  # 返回一模一样的代码
    harness = Harness(llm, sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py", max_attempts=2)

    assert report.status == "failed"
    assert any("未变化" in a.error for a in report.attempts)
    assert len(sandbox.calls) == 1  # 第 2 次没执行


async def test_harness_blocks_bad_patch(tmp_path):
    """LLM 修复出的代码带 pip install → Patch Policy 拒绝执行。"""
    ws = make_workspace(tmp_path, "print('bad')")
    sandbox = FakeSandbox([
        SandboxResult(stderr="error: first", exit_code=1),
    ])
    llm = FakeLLM(responses=["os.system('pip install torch')"])
    harness = Harness(llm, sandbox)  # type: ignore[arg-type]

    # max_attempts=2：attempt1 失败 → 修复出带毒补丁 → attempt2 被 Patch Policy 拒绝 → 耗尽
    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py", max_attempts=2)

    assert report.status == "failed"
    assert any("Patch Policy" in a.error for a in report.attempts)
    assert len(sandbox.calls) == 1  # 带毒补丁没被执行


async def test_harness_catches_metric_fraud(tmp_path):
    """exit_code=0 但 metrics.json 谎报 → 指标重算拦截，判失败。"""
    ws = make_workspace(tmp_path, "print('run with fraud metrics')")
    (Path(ws) / "metrics.json").write_text(
        json.dumps({"accuracy": 0.99}), encoding="utf-8"
    )
    (Path(ws) / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in [
            {"y_true": "a", "y_pred": "a"},
            {"y_true": "b", "y_pred": "a"},  # 真实 accuracy 只有 0.5
        ]),
        encoding="utf-8",
    )
    sandbox = FakeSandbox([SandboxResult(stdout="done", exit_code=0)])
    llm = FakeLLM(responses=["print('patched')"])
    harness = Harness(llm, sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(ws, "fake-sandbox", "run.py", max_attempts=2)

    assert report.status == "failed"
    assert any("疑似伪造" in a.error for a in report.attempts)


async def test_harness_missing_entrypoint(tmp_path):
    """入口文件不存在：直接失败，不起容器。"""
    sandbox = FakeSandbox([])
    harness = Harness(FakeLLM(), sandbox)  # type: ignore[arg-type]

    report = await harness.run_with_healing(str(tmp_path), "fake-sandbox", "run.py")

    assert report.status == "failed"
    assert "不存在" in report.reason
    assert sandbox.calls == []
