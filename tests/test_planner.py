"""测试 planner：模板生成 + 校验器（产物契约/高契约）+ LLM 优先模板兜底。"""

from types import SimpleNamespace

from tests.conftest import FakeLLM, make_step
from scholar_agent.agent.planner.planner import build_plan
from scholar_agent.agent.planner.templates import build_steps_from_template
from scholar_agent.agent.planner.validator import (
    validate_critical_contracts,
    validate_steps,
)
from scholar_agent.models.intent import IntentContext, IntentType


def make_intent(itype=IntentType.paper_reproduction) -> IntentContext:
    return IntentContext(
        raw_intent="复现 Attention",
        rewritten_intent="复现 Attention Is All You Need",
        intent_type=itype,
        entities={"paper_title": "Attention Is All You Need"},
    )


# ─── 模板 ───

def test_paper_template_shape():
    """论文复现模板：4 步线性序列，repo_discovery 归确定性 Agent。"""
    steps = build_steps_from_template(make_intent())
    assert [s.type for s in steps] == [
        "paper_parse", "repo_discovery", "code_run", "report",
    ]
    repo_step = steps[1]
    assert repo_step.agent == "research_coding_agent"
    assert repo_step.inputs["paper_title"] == "Attention Is All You Need"


def test_template_passes_validation():
    """模板产物必须能过自家校验器（自洽性）。"""
    steps = build_steps_from_template(make_intent())
    ok, reason = validate_steps(steps)
    assert ok, reason
    ok, reason = validate_critical_contracts(steps)
    assert ok, reason


# ─── 校验器 ───

def test_validator_detects_missing_producer():
    """数据契约：消费了没人产出的 artifact 必须报错。"""
    s1 = make_step("s1", required=["ghost_artifact"])
    ok, reason = validate_steps([s1])
    assert not ok
    assert "没有" in reason


def test_validator_detects_duplicate_producer():
    """写冲突：两个步骤产出同名 artifact 必须报错。"""
    s1 = make_step("s1", outputs=["dup"])
    s2 = make_step("s2", outputs=["dup"])
    ok, reason = validate_steps([s1, s2])
    assert not ok
    assert "多个 producer" in reason


def test_validator_rejects_empty_plan():
    """空计划：直接拒绝（LLM 返回空列表的兜底）。"""
    ok, reason = validate_steps([])
    assert not ok
    assert "空" in reason


def test_critical_contract_blocks_llm_assignee():
    """高契约：repo_discovery 被 LLM 指派给别的 Agent 必须拒绝。"""
    bad = make_step("s1", type_="repo_discovery", agent="coder_agent")
    ok, reason = validate_critical_contracts([bad])
    assert not ok
    assert "research_coding_agent" in reason


# ─── planner 主入口 ───

def _blueprint_steps():
    """合法 LLM 蓝图（data_agent 出报告）。"""
    return SimpleNamespace(steps=[
        SimpleNamespace(ref="s1", name="解析", type="paper_parse",
                        agent="librarian_agent",
                        required_artifacts=[], output_artifacts=["parsed_paper"]),
        SimpleNamespace(ref="s2", name="报告", type="report",
                        agent="data_agent",
                        required_artifacts=["parsed_paper"], output_artifacts=["final_report"]),
    ])


async def test_build_plan_uses_llm_when_valid():
    """LLM 蓝图合法 → 直接采用 LLM 步骤。"""
    llm = FakeLLM(structured={"PlanBlueprint": _blueprint_steps()})
    plan = await build_plan(llm, make_intent())
    assert [s.id for s in plan.steps] == ["s1", "s2"]
    assert plan.id and plan.trace_id


async def test_build_plan_falls_back_to_template():
    """LLM 失败 → 回退模板（永远可用）。"""
    llm = FakeLLM(structured={"PlanBlueprint": RuntimeError("api down")})
    plan = await build_plan(llm, make_intent())
    assert [s.type for s in plan.steps] == [
        "paper_parse", "repo_discovery", "code_run", "report",
    ]


async def test_build_plan_rejects_contract_violating_llm_plan():
    """LLM 把高契约步骤指派错 → 拒绝并回退模板。"""
    bad = SimpleNamespace(steps=[
        SimpleNamespace(ref="s1", name="搜仓库", type="repo_discovery",
                        agent="coder_agent",
                        required_artifacts=[], output_artifacts=["repo_url"]),
    ])
    llm = FakeLLM(structured={"PlanBlueprint": bad})
    plan = await build_plan(llm, make_intent())
    # 回退后的模板里 repo_discovery 归 research_coding_agent
    repo_step = next(s for s in plan.steps if s.type == "repo_discovery")
    assert repo_step.agent == "research_coding_agent"


async def test_build_plan_rejects_broken_contract_llm_plan():
    """LLM 蓝图产物契约不闭合（消费没人产的 artifact）→ 拒绝并回退模板。"""
    broken = SimpleNamespace(steps=[
        SimpleNamespace(ref="s1", name="跑代码", type="code_run",
                        agent="coder_agent",
                        required_artifacts=["ghost"], output_artifacts=["run_result"]),
    ])
    llm = FakeLLM(structured={"PlanBlueprint": broken})
    plan = await build_plan(llm, make_intent())
    # 回退后的模板是完整 4 步链
    assert [s.type for s in plan.steps] == [
        "paper_parse", "repo_discovery", "code_run", "report",
    ]
