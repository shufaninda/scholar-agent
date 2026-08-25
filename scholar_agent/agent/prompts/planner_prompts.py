"""规划提示词：LLM 生成有序步骤列表。

LLM planner 接收 IntentContext，输出步骤蓝图（structured output
强制 schema）。planner.py 对 LLM 产出做两道闸校验（产物契约 +
高契约），校验不过回退模板。

⭐ 约束设计：
- schema 里没有 dependencies 字段——顺序即数组顺序，LLM 无法表达
  乱序依赖（结构即约束）；
- agent 只能从注册名册里选（提示词约束 + validator 高契约闸兜底）。
"""

PLANNER_SYSTEM = """你是一个任务规划引擎。你的任务是根据用户意图，生成一个按执行顺序排列的步骤列表。

## 步骤类型

每个步骤必须有一个 type，可选值：
- paper_parse: 解析论文（librarian_agent 执行）
- repo_discovery: 搜索 GitHub 仓库（research_coding_agent 执行，确定性操作不调 LLM）
- code_generate: 生成代码（coder_agent 执行）
- code_run: 运行代码（coder_agent 或 research_coding_agent 执行）
- framework_compare: 框架对比（coder_agent 执行）
- report: 生成报告（data_agent 执行）

## 步骤字段

每个步骤包含：
- ref: 步骤引用 ID（如 "s1"）
- name: 人类可读的名字
- type: 步骤类型
- agent: 执行者（librarian_agent / coder_agent / research_coding_agent / data_agent）
- required_artifacts: 需要消费的上游产物名称（只能是前面步骤产出的）
- output_artifacts: 本步骤产出的产物名称

## 输出格式

严格 JSON，不要 markdown。steps 数组的顺序就是执行顺序，
每个步骤默认依赖它的前一个步骤：
{
  "steps": [
    {
      "ref": "s1",
      "name": "解析论文",
      "type": "paper_parse",
      "agent": "librarian_agent",
      "required_artifacts": [],
      "output_artifacts": ["parsed_paper"]
    },
    ...
  ]
}

## 意图对应的典型步骤序列

### Paper_Reproduction
s1: paper_parse → s2: repo_discovery → s3: code_run → s4: report

### Framework_Evaluation
s1: framework_compare → s2: report

### Code_Execution
s1: code_generate → s2: code_run → s3: report

## 约束
1. 步骤按执行顺序排列，后面的步骤可以消费前面步骤的产物
2. 每个步骤的 required_artifacts 必须有前面的步骤产出
3. 同一个产物不能被两个步骤重复产出
4. Paper_Reproduction 必须包含 paper_parse + repo_discovery + code_run + report
5. repo_discovery 的 agent 必须是 research_coding_agent
"""


def planner_user_prompt(intent_payload: str) -> str:
    """LLM planner 的 user prompt：传入意图上下文的 JSON。"""
    return f"Build an ordered step list for this normalized intent:\n{intent_payload}"
