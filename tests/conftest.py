"""公共 fixture：fakeredis / FakeLLM / 沙箱与步骤构造工厂。

测试原则：不联网、不花钱、不起真容器。
- fake_redis：fakeredis 替代真 Redis
- fake_llm：预设返回值的假 LLM（duck typing 兼容 LLMClient 接口）
- make_step / make_plan：快速构造 Step / Plan
"""

from types import SimpleNamespace

import fakeredis.aioredis
import pytest

from scholar_agent.models.plan import Plan
from scholar_agent.models.step import Step


@pytest.fixture
async def fake_redis():
    """fakeredis 异步客户端，模拟 decode_responses=True 行为。"""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


class FakeStructured:
    """with_structured_output 返回的假 Runnable：ainvoke 返回预设 schema 实例。"""

    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error

    async def ainvoke(self, prompt, **kwargs):
        if self.error:
            raise self.error
        return self.value


class FakeLLM:
    """测试用假 LLM：按调用顺序弹出预设响应，记录所有调用。

    - responses: list[str | Exception]，ainvoke 按序消费；耗尽后返回空串
    - structured: dict[schema名 -> 返回值或Exception]，给 with_structured_output
    """

    def __init__(self, responses=None, structured=None):
        self.responses = list(responses or [])
        self.structured_map = structured or {}
        self.calls: list = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.responses:
            return SimpleNamespace(content="")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return SimpleNamespace(content=resp)

    def with_structured_output(self, schema):
        value = self.structured_map.get(schema.__name__)
        if isinstance(value, Exception):
            return FakeStructured(error=value)
        return FakeStructured(value=value)


@pytest.fixture
def fake_llm():
    return FakeLLM()


def make_step(
    step_id: str,
    name: str = "",
    type_: str = "code_run",
    agent: str = "coder_agent",
    required: list[str] | None = None,
    outputs: list[str] | None = None,
    inputs: dict | None = None,
) -> Step:
    """快速构造 Step（测试工厂）。"""
    return Step(
        id=step_id,
        name=name or step_id,
        type=type_,
        agent=agent,
        required_artifacts=required or [],
        output_artifacts=outputs or [],
        inputs=inputs or {},
    )


def make_plan(steps: list[Step]) -> Plan:
    """快速构造 Plan（测试工厂）。"""
    import uuid

    return Plan(id=str(uuid.uuid4()), steps=steps, user_intent="test")
