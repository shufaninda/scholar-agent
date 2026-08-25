"""DataAgent：生成最终报告。

输入：Plan.artifacts 里的全部上游产物（run_result / comparison_report 等）
输出：final_report artifact（结构化 Markdown 报告）
"""

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.worker_prompts import DATA_SYSTEM, data_user_prompt
from scholar_agent.agent.workers.base import BaseAgent
from scholar_agent.models.artifact import Artifact, ArtifactType
from scholar_agent.models.step import Step

MAX_ARTIFACT_CHARS = 2000  # 单个 artifact 截断，防 prompt 爆长度


class DataAgent(BaseAgent):
    """报告生成 Agent：收集所有 artifact → LLM → Markdown 报告。"""

    name = "data_agent"

    def __init__(self, llm: LLMClient):
        super().__init__(llm=llm)

    async def execute(
        self, step: Step, artifacts: dict[str, Artifact]
    ) -> dict[str, Artifact]:
        """汇总 artifacts → final_report artifact。"""
        material = self._collect_material(artifacts)
        response = await self.llm.ainvoke([
            {"role": "system", "content": DATA_SYSTEM},
            {"role": "user", "content": data_user_prompt(material)},
        ])

        return {
            "final_report": Artifact(
                key="final_report",
                type=ArtifactType.report,
                value=response.content,
                producer_task_id=step.id,
            )
        }

    def _collect_material(self, artifacts: dict[str, Artifact]) -> str:
        """把所有 artifact 拼成文本材料（截断防爆 prompt）。"""
        if not artifacts:
            return "（无上游产物，请基于任务描述生成报告）"
        sections = []
        for key, art in artifacts.items():
            value = str(art.value or "")[:MAX_ARTIFACT_CHARS]
            sections.append(f"### [{key}] (type={art.type.value})\n{value}")
        return "\n\n".join(sections)
