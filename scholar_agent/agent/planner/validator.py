"""步骤校验器：约束 LLM 规划产出的两道闸。

原五项校验中，"依赖节点存在"与"Kahn 检环"已删——步骤数组在结构上
不可能引用不存在的依赖（下标即依赖），也不可能成环（下标单向递增）。
这类错误由数据结构本身杜绝，比运行时校验更可靠。

留下的两道闸都是"约束 LLM"的核心：
    1. validate_steps：产物契约闭合（producer 存在 + 唯一）；
    2. validate_critical_contracts：高契约步骤必须派给确定性 Agent。
"""

from scholar_agent.models.step import Step

# 高契约映射：type → 必须的 agent。
# repo_discovery 涉及外部 URL 选择，LLM 会编造 URL，必须走确定性后端。
CRITICAL_CONTRACTS = {
    "repo_discovery": "research_coding_agent",
}


def validate_steps(steps: list[Step]) -> tuple[bool, str]:
    """校验步骤列表的产物契约（提前返回风格）。

    - 非空检查；
    - 同一 artifact 不能有多个 producer（写冲突检测）；
    - 每个步骤的 required_artifacts 必须有 producer。

    返回 (ok, reason)。reason 为空串表示校验通过。
    """
    if not steps:
        return False, "计划为空"

    producers = _collect_producers(steps)
    for key, owners in producers.items():
        if len(owners) > 1:
            return False, f"产物 {key} 有多个 producer: {owners}"

    for step in steps:
        for required in step.required_artifacts:
            if required not in producers:
                return False, f"步骤 {step.id} 消费的产物 {required} 没有产出方"
    return True, ""


def _collect_producers(steps: list[Step]) -> dict[str, list[str]]:
    """产物名 → producer 步骤 ID 列表（列表长度 > 1 即写冲突）。"""
    producers: dict[str, list[str]] = {}
    for step in steps:
        for key in step.output_artifacts:
            producers.setdefault(key, []).append(step.id)
    return producers


def validate_critical_contracts(steps: list[Step]) -> tuple[bool, str]:
    """校验高契约步骤：type 在 CRITICAL_CONTRACTS 里的必须派给指定 Agent。

    典型场景：LLM 把 repo_discovery 派给 coder_agent——coder 会用
    LLM 编造 URL，破坏确定性。校验不过则整份 LLM 规划作废回退模板。
    """
    for step in steps:
        required_agent = CRITICAL_CONTRACTS.get(step.type)
        if required_agent and step.agent != required_agent:
            return False, (
                f"步骤 {step.id}({step.type}) 必须由 {required_agent} 执行，"
                f"LLM 指派了 {step.agent}"
            )
    return True, ""
