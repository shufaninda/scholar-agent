"""规划主入口：LLM 优先 + 模板兜底。

策略（fail-safe 链）：
    LLM 规划 → 两道闸校验（产物契约 + 高契约）→ 校验不过 → 模板兜底
模板是确定性的、永远可用——LLM 挂了系统照样能出计划。
"""

import logging
import uuid

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.planner import agent_planner
from scholar_agent.agent.planner.templates import build_steps_from_template
from scholar_agent.agent.planner.validator import (
    validate_critical_contracts,
    validate_steps,
)
from scholar_agent.models.intent import IntentContext
from scholar_agent.models.plan import Plan
from scholar_agent.models.step import Step

logger = logging.getLogger(__name__)


async def build_plan(llm: LLMClient, intent: IntentContext) -> Plan:
    """规划主入口：先试 LLM，任何失败回退模板。"""
    steps = await _try_llm_plan(llm, intent)
    if steps is None:
        steps = build_steps_from_template(intent)
        logger.info("planner_fallback_to_template intent=%s", intent.intent_type)

    return Plan(
        id=str(uuid.uuid4()),
        trace_id=f"trace_{uuid.uuid4().hex[:8]}",
        user_intent=intent.rewritten_intent or intent.raw_intent,
        intent_type=intent.intent_type.value,
        steps=steps,
    )


async def _try_llm_plan(llm: LLMClient, intent: IntentContext) -> list[Step] | None:
    """尝试 LLM 规划 + 两道闸校验。任何一步失败返回 None（提前返回）。"""
    try:
        steps = await agent_planner.build_steps(llm, intent)
    except Exception as exc:  # noqa: BLE001 LLM 边界统一兜底
        logger.warning("llm_planner_failed: %s", exc)
        return None

    ok, reason = validate_critical_contracts(steps)
    if not ok:
        logger.warning("llm_plan_contract_violation: %s", reason)
        return None

    ok, reason = validate_steps(steps)
    if not ok:
        logger.warning("llm_plan_invalid: %s", reason)
        return None

    return steps
