"""LLM 规划 Agent：让 LLM 生成步骤蓝图（blueprint）。

流程：拼 prompt → 调 LLM（structured output 强制 schema）→ 转换为
list[Step]。任何一步失败都抛异常，由上层 planner.py 回退模板。

注意：蓝图 schema 里没有 dependencies 字段——LLM 想输出依赖都会被
schema 拒绝。执行顺序由数组下标表达（"代码约束 LLM"的极致形态）。
"""

import json
import logging

from pydantic import BaseModel, Field

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.prompts.planner_prompts import (
    PLANNER_SYSTEM,
    planner_user_prompt,
)
from scholar_agent.models.intent import IntentContext
from scholar_agent.models.step import Step

logger = logging.getLogger(__name__)


class StepBlueprint(BaseModel):
    """LLM 输出的单个步骤蓝图（ref 是 LLM 内部引用 ID）。"""

    ref: str
    name: str
    type: str
    agent: str
    required_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)


class PlanBlueprint(BaseModel):
    """LLM 输出的整体蓝图：{"steps": [...]}，顺序即执行顺序。"""

    steps: list[StepBlueprint] = Field(default_factory=list)


async def build_steps(llm: LLMClient, intent: IntentContext) -> list[Step]:
    """调 LLM 生成有序步骤列表。失败抛异常（上层回退模板）。"""
    blueprint = await _ask_llm_for_blueprint(llm, intent)
    # 蓝图的 ref 直接当步骤 ID 用（LLM 用 ref 自我引用，保持一致即可）
    return [
        Step(
            id=bp.ref,
            name=bp.name,
            type=bp.type,
            agent=bp.agent,
            required_artifacts=bp.required_artifacts,
            output_artifacts=bp.output_artifacts,
            inputs=_extract_step_inputs(intent, bp),
        )
        for bp in blueprint.steps
    ]


async def _ask_llm_for_blueprint(llm: LLMClient, intent: IntentContext) -> PlanBlueprint:
    """调 LLM 拿结构化蓝图（Structured Outputs 强制 JSON schema）。"""
    intent_payload = json.dumps(intent.model_dump(mode="json"), ensure_ascii=False)
    structured = llm.with_structured_output(PlanBlueprint)
    result = await structured.ainvoke([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": planner_user_prompt(intent_payload)},
    ])
    if not result.steps:
        raise ValueError("LLM 规划返回空步骤列表")
    return result


def _extract_step_inputs(intent: IntentContext, bp: StepBlueprint) -> dict:
    """把意图实体透传给需要的步骤（repo_discovery 要 paper_title）。

    ⭐ inputs 由代码填（不走 LLM）：实体提取已在意图阶段由代码完成，
    规划阶段只做确定性透传——LLM 编造的输入不可信。
    """
    if bp.type != "repo_discovery":
        return {}
    paper_title = intent.entities.get("paper_title") or intent.rewritten_intent
    return {"paper_title": paper_title}
