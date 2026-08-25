"""Agent 基类：定义统一执行接口。

约定：所有 Agent 的 execute 接收 (step, artifacts)，返回"要合并进
Plan.artifacts 的新产物 dict"。不直接改 state——那是 execute_step
节点（graph.py）的职责。
"""

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox
from scholar_agent.models.artifact import Artifact
from scholar_agent.models.step import Step


class BaseAgent:
    """Agent 基类：name + 依赖注入（llm / sandbox）。"""

    name: str = "base_agent"

    def __init__(self, llm: LLMClient | None = None, sandbox: DockerSandbox | None = None):
        self.llm = llm
        self.sandbox = sandbox

    async def execute(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """执行步骤，返回要更新到产物仓库的新产物。"""
        raise NotImplementedError
