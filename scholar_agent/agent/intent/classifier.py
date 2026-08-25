"""意图分类器：三路并行（分类 + 重写 + 论文字段抽取）。

三路并行设计：
    路 A：ClassifyOnly         → LLM 返回 intent_type + entities + confidence
    路 B：Rewrite              → LLM 把用户 query 重写为专业表述
    路 C：ExtractPaperFields   → LLM 提取 paper_title / arxiv_id / method_name

降级语义：
    - classify 失败 → 致命，整体返回 error
    - rewrite 失败  → 非致命，降级用原 query
    - extract 失败  → 非致命，降级用空 dict

用 asyncio.gather(return_exceptions=True) 并行编排三路：
    - return_exceptions=True 让异常不抛出，而是作为结果返回
    - 调用方用 isinstance(result, Exception) 判断是否失败
    - 这样三路互不影响，可以独立降级

数据流：
    用户输入 → rule_router 预分类（命中就直接返回，不调 LLM）
           → 没命中 → classifier 三路并行调 LLM
           → 返回 IntentContext
"""

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.agent.intent.memory import IntentMemoryStore
from scholar_agent.agent.intent.rule_router import route_by_rule
from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.models.intent import IntentContext, IntentType, PaperSearchFields

logger = logging.getLogger(__name__)


class IntentClassifier:
    """意图分类器：三路并行 + 优雅降级。

    用法：
        classifier = IntentClassifier(llm, memory_store)
        intent = await classifier.classify("复现 Attention 论文", session_id="xxx")
        # intent 是 IntentContext
    """

    def __init__(self, llm: LLMClient, memory: IntentMemoryStore):
        self.llm = llm
        self.memory = memory
        self._bg_tasks: set[asyncio.Task] = set()  # 后台写记忆任务的强引用

    async def classify(
        self, raw_query: str, session_id: str = ""
    ) -> IntentContext:
        """意图识别主入口。

        流程：
            1. 规则预分类（命中就直接返回，不调 LLM）
            2. 没命中 → 加载 Redis 记忆 + 三路并行调 LLM
            3. 降级处理（classify 致命，rewrite/extract 降级）
            4. 异步写入 Redis 记忆

        Args:
            raw_query: 用户原始输入
            session_id: 会话 ID（用于 Redis 记忆，可为空）

        Returns:
            IntentContext（包含意图类型 + 重写后 query + 实体）
        """
        # ─── 第 1 步：规则预分类（快速路径）───
        rule_result = route_by_rule(raw_query)
        if rule_result is not None:
            # 命中规则，直接返回（不调 LLM，省时间省 token）
            return IntentContext(
                raw_intent=raw_query,
                rewritten_intent=raw_query,
                intent_type=rule_result,
                source="rule",
            )

        # ─── 第 2 步：加载 Redis 记忆 ───
        memory_turns = await self.memory.load_recent_turns(session_id)

        # ─── 第 3 步：三路并行调 LLM ───
        # ⭐ 三路并行 + 独立降级
        classify_r, rewrite_r, extract_r = await asyncio.gather(
            self._classify_only(raw_query, memory_turns),
            self._rewrite(raw_query, memory_turns),
            self._extract_paper_fields(raw_query, memory_turns),
            return_exceptions=True,
        )

        # ─── 第 4 步：降级处理 ───
        # classify 失败 → 致命，返回 unknown（intent_node 会 Command 到 END）
        if isinstance(classify_r, Exception):
            logger.warning("意图分类失败: %s", classify_r)
            return IntentContext(
                raw_intent=raw_query,
                intent_type=IntentType.unknown,
                source="llm_failed",
            )

        # rewrite 失败 → 非致命，降级用原 query
        rewritten = raw_query if isinstance(rewrite_r, Exception) else rewrite_r

        # extract 失败 → 非致命，降级用空 dict
        entities = dict(classify_r.get("entities", {}))
        if not isinstance(extract_r, Exception):
            entities.update(extract_r)

        # ─── 第 5 步：异步写入记忆（不阻塞返回）───
        # 裸 create_task 只有弱引用会被 GC 静默回收，存强引用集兜底
        self._bg_tasks.add(
            task := asyncio.create_task(
                self.memory.append_turn(session_id, "user", raw_query)
            )
        )
        task.add_done_callback(self._bg_tasks.discard)

        return IntentContext(
            raw_intent=raw_query,
            rewritten_intent=rewritten,
            intent_type=classify_r["intent_type"],
            entities=entities,
            confidence=classify_r.get("confidence", 0.0),
            source="llm",
        )

    # ──────────────────────────────────────────────
    # 三路并行的三个子任务
    # ──────────────────────────────────────────────

    async def _classify_only(
        self, query: str, memory: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """路 A：LLM 意图分类（Structured Outputs 强制 JSON）。"""
        from scholar_agent.agent.prompts.intent_prompts import (
            CLASSIFY_SYSTEM,
            classify_user_prompt,
        )

        class ClassifyResult(BaseModel):
            """路 A 的结构化返回 schema。"""
            intent_type: IntentType
            entities: dict[str, Any] = Field(default_factory=dict)
            confidence: float = Field(0.0, ge=0, le=1)

        structured = self.llm.with_structured_output(ClassifyResult)
        result = await structured.ainvoke([
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": classify_user_prompt(query, json.dumps(memory, ensure_ascii=False))},
        ])
        return {
            "intent_type": result.intent_type,
            "entities": result.entities,
            "confidence": result.confidence,
        }

    async def _rewrite(
        self, query: str, memory: list[dict[str, Any]]
    ) -> str:
        """路 B：LLM query 重写为专业表述。"""
        from scholar_agent.agent.prompts.intent_prompts import (
            REWRITE_SYSTEM,
            rewrite_user_prompt,
        )

        resp = await self.llm.ainvoke([
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": rewrite_user_prompt(query, json.dumps(memory, ensure_ascii=False))},
        ])
        return str(resp.content).strip()

    async def _extract_paper_fields(
        self, query: str, memory: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """路 C：LLM 论文字段抽取（Structured Outputs → PaperSearchFields）。"""
        from scholar_agent.agent.prompts.intent_prompts import (
            EXTRACT_SYSTEM,
            extract_user_prompt,
        )

        structured = self.llm.with_structured_output(PaperSearchFields)
        result = await structured.ainvoke([
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_user_prompt(query)},
        ])
        return result.model_dump()
