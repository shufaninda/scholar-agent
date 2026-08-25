"""CoderAgent：生成代码 + 沙箱执行。

任务类型分发：
    code_generate:      LLM 生成代码 → generated_code artifact
    code_run:           消费 generated_code → 一次性沙箱执行 → run_result artifact
    framework_compare:  LLM 生成对比实验代码 → 沙箱执行 → comparison_report artifact
"""

import logging

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.worker_prompts import CODER_SYSTEM, coder_user_prompt
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.agent.workers.base import BaseAgent
from scholar_agent.models.artifact import Artifact, ArtifactType
from scholar_agent.models.step import Step

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """代码 Agent：按 step.type 分发生成/执行。"""

    name = "coder_agent"

    def __init__(self, llm: LLMClient, sandbox: DockerSandbox):
        super().__init__(llm=llm, sandbox=sandbox)

    async def execute(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """按步骤类型分发。"""
        if step.type == "code_generate":
            return await self._generate_code(step)
        if step.type == "code_run":
            return await self._run_code(step, artifacts)
        if step.type == "framework_compare":
            return await self._compare_frameworks(step)
        raise ValueError(f"CoderAgent 不支持的步骤类型: {step.type}")

    async def _generate_code(self, step: Step) -> dict[str, Artifact]:
        """code_generate：LLM 生成代码（不执行）。"""
        code = await self._ask_llm_for_code(step.description or step.name)
        return {
            "generated_code": Artifact(
                key="generated_code",
                type=ArtifactType.code,
                value=code,
                producer_task_id=step.id,
            )
        }

    async def _run_code(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """code_run：一次性沙箱执行 generated_code（断网容器）。"""
        if "generated_code" not in artifacts:
            raise ValueError("code_run 缺少上游产物 generated_code")
        code = str(artifacts["generated_code"].value)

        result = await self.sandbox.execute(code, timeout=step.timeout_seconds)
        output = result.stdout if result.ok else (
            f"[exit_code={result.exit_code}]\n{result.stderr or result.error or result.stdout}"
        )
        return {
            "run_result": Artifact(
                key="run_result",
                type=ArtifactType.text,
                value=output,
                producer_task_id=step.id,
                metadata={"exit_code": result.exit_code, "ok": result.ok},
            )
        }

    async def _compare_frameworks(self, step: Step) -> dict[str, Artifact]:
        """framework_compare：生成对比代码并执行，产出对比报告。"""
        code = await self._ask_llm_for_code(
            f"框架对比实验：{step.description or step.name}。"
            "生成对比代码，把对比结果 print 输出。"
        )
        result = await self.sandbox.execute(code, timeout=step.timeout_seconds)
        output = result.stdout if result.ok else (
            f"[exit_code={result.exit_code}]\n{result.stderr or result.error or result.stdout}"
        )
        return {
            "comparison_report": Artifact(
                key="comparison_report",
                type=ArtifactType.text,
                value=output,
                producer_task_id=step.id,
                metadata={"exit_code": result.exit_code, "ok": result.ok},
            )
        }

    async def _ask_llm_for_code(self, description: str) -> str:
        """调 LLM 生成纯代码文本。"""
        response = await self.llm.ainvoke([
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": coder_user_prompt(description)},
        ])
        return response.content
