"""LibrarianAgent：解析论文，提取方法/数据集/关键参数。

输入：step.inputs["paper_title"]（planner 从意图 entities 填充）
输出：parsed_paper artifact（结构化 Markdown 论文分析报告）
下游：repo_discovery 步骤消费它构造 GitHub 搜索词。
"""

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.worker_prompts import LIBRARIAN_SYSTEM
from scholar_agent.agent.workers.base import BaseAgent
from scholar_agent.models.artifact import Artifact, ArtifactType
from scholar_agent.models.step import Step


class LibrarianAgent(BaseAgent):
    """论文解析 Agent：调 LLM 产出复现分析报告。"""

    name = "librarian_agent"

    def __init__(self, llm: LLMClient):
        super().__init__(llm=llm)

    async def execute(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """解析论文 → parsed_paper artifact。"""
        paper_title = step.inputs.get("paper_title") or step.description or "未知论文"
        method = step.inputs.get("method_name", "")
        extra = f"\n论文方法名：{method}" if method else ""

        response = await self.llm.ainvoke([
            {"role": "system", "content": LIBRARIAN_SYSTEM},
            {"role": "user", "content": f"请解析论文：《{paper_title}》{extra}"},
        ])

        return {
            "parsed_paper": Artifact(
                key="parsed_paper",
                type=ArtifactType.text,
                value=response.content,
                producer_task_id=step.id,
            )
        }
