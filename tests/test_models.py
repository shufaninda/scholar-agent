"""测试 models：Step 状态机默认值 + Plan 序列化往返。"""

from tests.conftest import make_plan, make_step


def test_step_defaults():
    """新建步骤默认 pending、无产物契约。"""
    step = make_step("s1")
    assert step.status.value == "pending"
    assert step.required_artifacts == []
    assert step.output_artifacts == []
    assert step.error is None


def test_plan_serialization_roundtrip():
    """Plan 能 JSON 序列化 + 反序列化（SSE/API 传输基础）。"""
    from scholar_agent.models.plan import Plan

    plan = make_plan([make_step("s1", outputs=["parsed_paper"])])
    data = plan.model_dump(mode="json")
    restored = Plan.model_validate(data)
    assert restored.steps[0].id == "s1"
    assert restored.user_intent == "test"
    assert restored.status.value == "pending"
