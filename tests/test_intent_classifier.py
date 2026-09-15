"""测试意图分类器：规则快速路径 + 三路并行 + 各路独立降级。"""

from types import SimpleNamespace

from scholar_agent.agent.intent.classifier import IntentClassifier
from scholar_agent.agent.intent.memory import IntentMemoryStore
from scholar_agent.models.intent import IntentType, PaperSearchFields
from tests.conftest import FakeLLM


def make_classifier(fake_llm, redis=None) -> IntentClassifier:
    return IntentClassifier(fake_llm, IntentMemoryStore(redis))


async def test_rule_fast_path_skips_llm():
    """规则命中直接返回，不调 LLM（省 token）。"""
    llm = FakeLLM()
    ctx = await make_classifier(llm).classify("复现 Attention Is All You Need")
    assert ctx.intent_type == IntentType.paper_reproduction
    assert ctx.source == "rule"
    assert llm.calls == []  # LLM 一次都没被调


async def test_three_way_parallel_happy_path():
    """三路都成功：分类 + 重写 + 抽取合并。"""
    llm = FakeLLM(
        responses=["复现 Attention 论文的完整实验"],
        structured={
            # 路 A：ClassifyResult 在函数内定义，schema 名一致
            "ClassifyResult": SimpleNamespace(
                intent_type=IntentType.paper_reproduction,
                entities={"topic": "attention"},
                confidence=0.9,
            ),
            # 路 C：PaperSearchFields
            "PaperSearchFields": PaperSearchFields(
                paper_title="Attention Is All You Need",
                paper_search_query="attention is all you need",
            ),
        },
    )
    ctx = await make_classifier(llm).classify("帮我分析这个数据集的统计特性")
    assert ctx.intent_type == IntentType.paper_reproduction
    assert ctx.source == "llm"
    assert "Attention" in ctx.rewritten_intent
    assert ctx.entities["paper_title"] == "Attention Is All You Need"
    assert ctx.confidence == 0.9


async def test_classify_failure_is_fatal():
    """classify 失败 → 致命：返回 unknown，路由到 END。"""
    llm = FakeLLM(
        responses=[RuntimeError("rewrite ok")],  # 路 B 正常
        structured={
            "ClassifyResult": RuntimeError("llm down"),  # 路 A 失败
            "PaperSearchFields": RuntimeError("nope"),    # 路 C 失败
        },
    )
    ctx = await make_classifier(llm).classify("the statistical property of this dataset")
    assert ctx.intent_type == IntentType.unknown
    assert ctx.source == "llm_failed"


async def test_rewrite_failure_degrades_to_raw():
    """rewrite 失败 → 非致命：降级用原始 query。"""
    llm = FakeLLM(
        responses=[RuntimeError("rewrite failed")],
        structured={
            "ClassifyResult": SimpleNamespace(
                intent_type=IntentType.code_execution, entities={}, confidence=0.8
            ),
            "PaperSearchFields": RuntimeError("skip"),
        },
    )
    ctx = await make_classifier(llm).classify("the statistical property of this dataset")
    assert ctx.rewritten_intent == "the statistical property of this dataset"


async def test_extract_failure_degrades_to_empty():
    """extract 失败 → 非致命：entities 只含路 A 的部分。"""
    llm = FakeLLM(
        responses=["重写后的 query"],
        structured={
            "ClassifyResult": SimpleNamespace(
                intent_type=IntentType.framework_evaluation,
                entities={"frameworks": ["langchain"]},
                confidence=0.7,
            ),
            "PaperSearchFields": RuntimeError("extract failed"),
        },
    )
    ctx = await make_classifier(llm).classify("the statistical property of this dataset")
    assert ctx.entities == {"frameworks": ["langchain"]}
