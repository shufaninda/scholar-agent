"""LLM 调用封装：统一 LLM 客户端，支持 Structured Outputs。

所有需要调 LLM 的地方都通过这里，方便切换模型/供应商。
用 langchain-openai 的 ChatOpenAI，兼容 DeepSeek / OpenAI / 任何 OpenAI 兼容 API。

为什么需要封装：
    1. 统一配置：API key / base_url / model 从 config 读，不散落各处
    2. 统一接口：所有 Agent 调 self.llm.ainvoke()，不关心底层是 DeepSeek 还是 OpenAI
    3. 支持 Structured Outputs：with_structured_output 强制 LLM 返回结构化 JSON
       （intent 的 _extract_paper_fields 用这个强制返回 PaperSearchFields）

用法：
    llm = LLMClient()
    response = await llm.ainvoke("总结这篇论文")
    text = response.content

    # Structured Outputs（强制返回结构化数据）
    structured = llm.with_structured_output(PaperSearchFields)
    result = await structured.ainvoke("从用户输入提取论文字段")
    # result 是 PaperSearchFields 实例
"""

from langchain_openai import ChatOpenAI

from scholar_agent.core.config import settings


class LLMClient:
    """LLM 客户端封装：统一 LLM 调用入口。

    封装 langchain-openai 的 ChatOpenAI，兼容 DeepSeek（通过 base_url）。
    所有 Agent 通过这个类调 LLM，不直接实例化 ChatOpenAI。

    用法：
        llm = LLMClient()
        response = await llm.ainvoke("你好")
        print(response.content)

        # Structured Outputs
        structured = llm.with_structured_output(PaperSearchFields)
        result = await structured.ainvoke("提取论文字段")
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """初始化 LLM 客户端。

        Args:
            model: 模型名，默认从 config 读（deepseek-chat）
            temperature: 温度，默认从 config 读（0.3）
        """
        self.llm = ChatOpenAI(
            model=model or settings.LLM_MODEL,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

    async def ainvoke(self, messages):
        """异步调用 LLM。

        Args:
            messages: 可以是 str（简单文本）或 list[dict]（多轮对话）
                      如 "你好" 或 [{"role": "system", "content": "..."}, ...]

        Returns:
            AIMessage，.content 是 LLM 返回的文本
        """
        return await self.llm.ainvoke(messages)

    def with_structured_output(self, schema):
        """返回一个支持 Structured Outputs 的 LLM 实例。

        强制 LLM 返回指定 Pydantic schema 的结构化数据。
        用于 intent 的 _extract_paper_fields（强制返回 PaperSearchFields）。

        Args:
            schema: Pydantic BaseModel 类，如 PaperSearchFields

        Returns:
            Runnable，调 .ainvoke() 返回 schema 实例

        用法：
            structured = llm.with_structured_output(PaperSearchFields)
            result = await structured.ainvoke("复现 Attention Is All You Need")
            # result 是 PaperSearchFields 实例
            # result.paper_title == "Attention Is All You Need"
        """
        return self.llm.with_structured_output(schema)
