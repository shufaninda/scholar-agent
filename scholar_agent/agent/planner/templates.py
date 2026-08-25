"""步骤模板：4 种意图类型的固定步骤序列。

定位：LLM 规划失败/校验不过时的兜底（确定性、永远可用）。
模板只定义步骤内容，顺序由数组下标表达（无 deps、无 edges）。
"""

from scholar_agent.models.intent import IntentContext, IntentType
from scholar_agent.models.step import Step


def _step(
    sid: str,
    name: str,
    type_: str,
    agent: str,
    outputs: list[str],
    required: list[str] | None = None,
    inputs: dict | None = None,
) -> Step:
    """构造模板步骤（测试 + 模板共用的小工厂）。"""
    return Step(
        id=sid,
        name=name,
        type=type_,
        agent=agent,
        required_artifacts=required or [],
        output_artifacts=outputs,
        inputs=inputs or {},
    )


def build_paper_reproduction_steps(intent: IntentContext) -> list[Step]:
    """论文复现模板：解析论文 → 搜 repo → 跑代码 → 报告。

    ⭐ 高契约步骤 repo_discovery 固定 agent=research_coding_agent，
    走确定性后端（PyGithub + git），绕过 LLM。
    """
    paper_title = intent.entities.get("paper_title") or intent.rewritten_intent
    return [
        _step("s1", "解析论文", "paper_parse", "librarian_agent", ["parsed_paper"]),
        _step("s2", "搜索复现仓库", "repo_discovery", "research_coding_agent",
              ["repo_url"], ["parsed_paper"], {"paper_title": paper_title}),
        _step("s3", "运行复现代码", "code_run", "research_coding_agent",
              ["run_result"], ["repo_url"]),
        _step("s4", "生成报告", "report", "data_agent",
              ["final_report"], ["run_result"]),
    ]


def build_framework_evaluation_steps(intent: IntentContext) -> list[Step]:
    """框架对比模板：对比实验 → 报告。"""
    return [
        _step("s1", "框架对比实验", "framework_compare", "coder_agent",
              ["comparison_report"]),
        _step("s2", "生成报告", "report", "data_agent",
              ["final_report"], ["comparison_report"]),
    ]


def build_code_execution_steps(intent: IntentContext) -> list[Step]:
    """代码执行模板：生成代码 → 运行 → 报告。"""
    return [
        _step("s1", "生成代码", "code_generate", "coder_agent", ["generated_code"]),
        _step("s2", "运行代码", "code_run", "coder_agent",
              ["run_result"], ["generated_code"]),
        _step("s3", "生成报告", "report", "data_agent",
              ["final_report"], ["run_result"]),
    ]


def build_general_steps(intent: IntentContext) -> list[Step]:
    """通用模板：单步骤直接出报告。"""
    return [
        _step("s1", "生成回答", "report", "data_agent", ["final_report"]),
    ]


# 意图类型 → 模板函数 的映射表
TEMPLATES = {
    IntentType.paper_reproduction: build_paper_reproduction_steps,
    IntentType.framework_evaluation: build_framework_evaluation_steps,
    IntentType.code_execution: build_code_execution_steps,
    IntentType.general: build_general_steps,
}


def build_steps_from_template(intent: IntentContext) -> list[Step]:
    """按意图类型选模板，未知意图降级为通用模板。"""
    builder = TEMPLATES.get(intent.intent_type, build_general_steps)
    return builder(intent)
