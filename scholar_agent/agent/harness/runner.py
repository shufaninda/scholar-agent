"""Harness 自愈引擎：3 次重试 + 5 个边界约束。

⭐ 这是项目灵魂——"模型负责建议，代码负责约束"。

主循环骨架：

    for attempt in 1..3:
        if attempt > 1 and 代码 SHA256 没变: continue
        if 补丁不合法（Patch Policy）: LLM 修复 + continue
        备份 + 写入 + 执行
        if 成功:
            if 执行期间偷改文件（Fingerprint）: LLM 修复 + continue
            if 指标造假（重算不匹配）: LLM 修复 + continue
            还原 + return 成功
        if 熔断（CircuitBreaker）: return 失败
        注入负反馈（FeedbackInjector）+ LLM 修复

关键设计点：
    1. 复用持久化沙箱（sandbox_id + exec_in），不起新容器
    2. 入口文件从 workspace 读取，LLM 修复的是这个文件的内容
    3. Fingerprint 对比时把入口文件加白名单（Harness 修复的就是它）
    4. CircuitBreaker window=2（≤ max_attempts=3），每 run 重置
"""

import json
import logging
from pathlib import Path

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
from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.harness_prompts import (
    REPAIR_SYSTEM,
    repair_user_prompt,
)
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.models.harness import HarnessAttempt, HarnessReport

logger = logging.getLogger(__name__)


class Harness:
    """生产级 Harness：3 次重试 + 5 个边界约束。

    ⭐ 关键：复用持久化沙箱（sandbox_id），不起新容器。
    前面 ResearchCodingAgent 已经 clone repo + 装依赖，Harness 必须在这个
    同一容器里跑入口文件，否则 workspace/依赖都丢。
    """

    def __init__(self, llm: LLMClient, sandbox: DockerSandbox):
        self.llm = llm
        self.sandbox = sandbox
        # window=2 ≤ max_attempts=3，每 run_with_healing 开头会重置
        self.breaker = CircuitBreaker(window=2)

    async def run_with_healing(
        self,
        workspace: str,
        sandbox_id: str,           # ⭐ 持久化沙箱 ID（复用）
        entrypoint: str,            # ⭐ 入口文件（如 run.py），在持久化容器里跑
        max_attempts: int = 3,
    ) -> HarnessReport:
        """自愈执行主循环：读入口文件 → 3 次重试 → 5 个边界机制校验。"""
        attempts: list[HarnessAttempt] = []
        self.breaker = CircuitBreaker(window=2)  # 每 run 重置，避免跨 run 串扰
        original_fingerprint = repo_fingerprint(workspace)

        # 读取入口文件作为"原代码"（LLM 修复的是这个文件的内容）
        entry_path = Path(workspace) / entrypoint
        if not entry_path.exists():
            return HarnessReport(
                status="failed", attempts=[],
                reason=f"入口文件不存在: {entrypoint}",
            )
        code = entry_path.read_text(encoding="utf-8")

        for attempt in range(1, max_attempts + 1):
            # 机制2a：检测 LLM 修复后代码是否真的变了（没变就跳过执行）
            if attempt > 1:
                if code_sha256(code) == attempts[-1].code_hash:
                    attempts.append(HarnessAttempt(
                        attempt=attempt, exit_code=-1,
                        error="LLM 修复后代码未变化，跳过执行",
                        repaired=True,
                    ))
                    continue

            # 机制1：补丁静态校验（防 LLM 越权）
            patch_ok, patch_reason = validate_patch(code)
            if not patch_ok:
                attempts.append(HarnessAttempt(
                    attempt=attempt, exit_code=-1, error=patch_reason,
                    repaired=attempt > 1,
                ))
                if attempt == max_attempts:
                    break
                code = await self._repair_code(code, patch_reason, attempts)
                continue

            # 备份 + 写入当前代码到入口文件
            backup = self._backup_files(workspace, entrypoint)
            entry_path.write_text(code, encoding="utf-8")

            # ⭐ 在持久化沙箱里执行入口文件（不起新容器）
            result = await self.sandbox.exec_in(
                sandbox_id, ["python", entrypoint]
            )

            # 成功判定（多重校验）
            if result.exit_code == 0:
                # 机制2b：执行期间未偷改源码（白名单：入口文件本身）
                after_fp = repo_fingerprint(workspace)
                unauthorized = [
                    p for p in detect_unauthorized_changes(original_fingerprint, after_fp)
                    if p != entrypoint
                ]
                if unauthorized:
                    error = f"执行期间偷改文件：{unauthorized}"
                    attempts.append(HarnessAttempt(
                        attempt=attempt, exit_code=-1, error=error,
                        repaired=attempt > 1,
                    ))
                    self._restore_files(workspace, backup)
                    if attempt == max_attempts:
                        break
                    code = await self._repair_code(code, error, attempts)
                    continue

                # 机制3：指标重算防伪（metrics.json + predictions.jsonl 都在才校验）
                metrics_path = Path(workspace) / "metrics.json"
                preds_path = Path(workspace) / "predictions.jsonl"
                if metrics_path.exists() and preds_path.exists():
                    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
                    recomputed = recompute_metrics(str(preds_path))
                    if "error" in recomputed:
                        # 空预测文件直接判失败，不进 verify
                        attempts.append(HarnessAttempt(
                            attempt=attempt, exit_code=-1,
                            error=recomputed["error"], repaired=attempt > 1,
                        ))
                        self._restore_files(workspace, backup)
                        if attempt == max_attempts:
                            break
                        code = await self._repair_code(code, recomputed["error"], attempts)
                        continue
                    metrics_ok, metrics_reason = verify_metrics(reported, recomputed)
                    if not metrics_ok:
                        attempts.append(HarnessAttempt(
                            attempt=attempt, exit_code=-1, error=metrics_reason,
                            repaired=attempt > 1,
                        ))
                        self._restore_files(workspace, backup)
                        if attempt == max_attempts:
                            break
                        code = await self._repair_code(code, metrics_reason, attempts)
                        continue

                # 全部校验通过：还原入口文件，返回成功
                self._restore_files(workspace, backup)
                return HarnessReport(
                    status="passed", attempts=attempts,
                    final_result=result.stdout,
                )

            # 失败处理
            attempts.append(HarnessAttempt(
                attempt=attempt,
                exit_code=result.exit_code,
                error=result.stderr or result.stdout or "unknown error",
                repaired=attempt > 1,
                code_hash=code_sha256(code),
            ))
            self._restore_files(workspace, backup)

            # 机制4：CircuitBreaker 熔断检测
            should_break, break_reason = self.breaker.should_break(
                f"python {entrypoint}", result.stderr or result.stdout or ""
            )
            if should_break:
                return HarnessReport(
                    status="failed", attempts=attempts, reason=break_reason,
                )

            if attempt == max_attempts:
                break

            # 机制5：注入负反馈调 LLM 修复
            code = await self._repair_code(
                code, result.stderr or result.stdout or "", attempts
            )

        return HarnessReport(
            status="failed", attempts=attempts,
            reason=f"failed after {max_attempts} attempts",
        )

    async def _repair_code(
        self,
        code: str,
        error: str,
        failed_attempts: list[HarnessAttempt],
    ) -> str:
        """调 LLM 修复代码，注入历史失败负反馈（机制5）。"""
        feedback = build_feedback_prompt(failed_attempts)
        system = f"{REPAIR_SYSTEM}\n\n{feedback}" if feedback else REPAIR_SYSTEM

        response = await self.llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": repair_user_prompt(code, error[-2000:])},
        ])
        return response.content

    def _backup_files(self, workspace: str, entrypoint: str) -> dict[str, str]:
        """备份入口文件内容（仅备份会被 Harness 修改的文件）。"""
        entry_path = Path(workspace) / entrypoint
        if entry_path.exists():
            return {entrypoint: entry_path.read_text(encoding="utf-8")}
        return {}

    def _restore_files(self, workspace: str, backup: dict[str, str]) -> None:
        """还原入口文件（保证下次从干净状态开始）。"""
        for rel, content in backup.items():
            (Path(workspace) / rel).write_text(content, encoding="utf-8")
