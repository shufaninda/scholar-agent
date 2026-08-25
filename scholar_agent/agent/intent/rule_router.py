"""规则预分类：关键词匹配快速识别意图类型。

对应 Sea 的 DetectIntentType（API 层兜底）。
对应 DESIGN.md 1 节"规则预分类"。

为什么需要：
    LLM 分类慢（1-3 秒）且可能失败。规则预分类用关键词匹配快速给初步判断，
    命中就直接返回，没命中再交给 LLM。这是"优雅降级"的第一层。

工作方式：
    1. 遍历 IntentType 的关键词表（编译好的正则）
    2. 命中任意关键词 → 返回对应 IntentType
    3. 全部不命中 → 返回 None（交给 LLM 分类）

为什么用正则（词边界）而不是 `in` 子串匹配：
    子串匹配会误伤——"code" 会命中 "decoder/encode"，"paper" 会命中 "newspaper"，
    "run" 会命中 "runner"。\b 词边界只匹配完整单词，解决前后缀粘连误伤。

注意：
    1. \b 词边界只对 ASCII 有效，中文没有空格分词，直接套 \b 会把
       "复现论文" 拆成两个词导致匹配失败 → 所以中文关键词不加词边界。
    2. 规则预分类不是必须的——返回 None 时 LLM 也能分类。
       它只是"快速路径优化"，不要为了覆盖所有情况而加太多关键词。
"""

import re

from scholar_agent.models.intent import IntentType


def _kw(keyword: str) -> str:
    """把关键词转成正则片段。

    英文关键词加 \b 词边界（只匹配完整单词，防 "decoder" 误中 "code"）。
    中文关键词保持子串匹配（中文无空格分词，\b 会把词拆开导致不匹配）。
    """
    if keyword.isascii():
        return rf"\b{re.escape(keyword)}\b"
    return re.escape(keyword)


# 每个 IntentType 的关键词表（小写匹配，命中任意一个即判定为该意图）
# 原始关键词表保持字符串可读，编译后的正则放 _PATTERNS
KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.paper_reproduction: [
        "复现", "论文", "paper", "reproduce", "reproduction",
        "attention", "transformer", "bert", "gpt", "resnet",
    ],
    IntentType.framework_evaluation: [
        "对比", "比较", "框架", "framework", "compare", "evaluation",
        "vs", "versus", "benchmark",
    ],
    IntentType.code_execution: [
        "执行", "运行", "跑", "run", "execute", "code",
        "python", "script",
    ],
}

# 编译一次，模块加载时完成，运行期直接复用（避免每次调用重复编译）
_PATTERNS: dict[IntentType, list[re.Pattern]] = {
    intent: [re.compile(_kw(kw)) for kw in kws]
    for intent, kws in KEYWORDS.items()
}


def route_by_rule(query: str) -> IntentType | None:
    """规则预分类：关键词匹配。

    Args:
        query: 用户原始输入

    Returns:
        命中关键词 → 对应 IntentType
        全部不命中 → None（交给 LLM）

    例子：
        >>> route_by_rule("复现 Attention Is All You Need")
        IntentType.paper_reproduction

        >>> route_by_rule("对比 LangChain 和 LlamaIndex")
        IntentType.framework_evaluation

        >>> route_by_rule("帮我写个贪心算法")
        None  # 没命中，交给 LLM

        >>> route_by_rule("分析 encoder-decoder 结构")  # 误伤已修复
        None  # "code" 不再误中 "encoder"，不会误判成代码执行
    """
    query_lower = query.lower()
    for intent_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                return intent_type
    return None
