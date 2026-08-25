"""意图上下文模型。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """意图类型枚举。

    Python 重构只保留 4 种（砍掉 AutoResearch / Custom_Benchmark）。

    ⭐ intent_node 用 intent_type == unknown 决定 Command 到 END。
    """

    paper_reproduction = "Paper_Reproduction"        # 论文复现
    framework_evaluation = "Framework_Evaluation"    # 框架对比
    code_execution = "Code_Execution"                # 代码执行
    general = "General"                              # 通用问答
    unknown = "Unknown"                              # 未识别（路由到 END）


class IntentContext(BaseModel):
    """意图识别结果。

    intent_node 产出，planner_node 消费。
    包含三路并行的结果：classify（intent_type+confidence）+ rewrite（rewritten_intent）
    + extract（entities 里的论文字段）。
    """

    raw_intent: str
    """用户原始输入。"""

    rewritten_intent: str = ""
    """LLM 重写后的专业表述。重写失败降级为 raw_intent。"""

    intent_type: IntentType = IntentType.unknown
    """意图类型。classify 失败时为 unknown，intent_node 路由到 END。"""

    entities: dict[str, Any] = Field(default_factory=dict)
    """抽取的实体，如 paper_title / arxiv_id / method_name。
    extract 失败时为空 dict，不阻断主流程。"""

    constraints: dict[str, Any] = Field(default_factory=dict)
    """用户约束，如"用 PyTorch" / "不超过 100 行"。"""

    confidence: float = Field(default=0.0, ge=0, le=1)
    """LLM 分类置信度，0~1。"""

    reasoning: str = ""
    """LLM 分类的理由。调试用。"""

    source: str = "llm"
    """来源："llm" 或 "rule_fallback"。LLM 失败降级为规则时标 rule_fallback。"""


class PaperSearchFields(BaseModel):
    """论文检索字段，用于 GitHub repo 搜索。

    intent_node 第三路 _extract_paper_fields 用 Structured Outputs 强制 LLM 返回。
    planner 把这些字段填进 Step.inputs，ResearchCodingAgent 用来搜 GitHub。
    """

    paper_title: str
    """论文英文标题。"""

    arxiv_id: str | None = None
    """arxiv 编号，如 "1706.03762"。"""

    paper_search_query: str
    """GitHub 搜索查询词，如 "attention is all you need transformer"。"""

    method_name: str | None = None
    """论文方法名，如 "Transformer"。"""

    confidence: float = Field(default=0.0, ge=0, le=1)
    """抽取置信度。"""
