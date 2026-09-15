"""规则快速路径样本覆盖速测：30 条典型指令 → 逐条命中一览。

背景：规则预分类是"意图识别的快速路径"——命中即免一次 LLM 调用。
本脚本把"自建 30 条典型指令样本"固化进仓库，一行命令复现覆盖率：

    uv run python scripts/rule_router_sample.py

最近一次实测：命中 23/30 ≈ 77%（未命中由 LLM 分类兜底，属设计冗余）。
已知取舍：快速路径覆盖优先、不做语义精判（如"推荐几篇论文"会命中复现类），
若要收紧可加负向规则或提高关键词门槛。
"""

from scholar_agent.agent.intent.rule_router import route_by_rule

# 典型指令样本（论文复现 10 + 框架评测 6 + 代码执行 5 + 通用/边界 9）
SAMPLES = [
    # ── 论文复现类 ──
    "复现 Attention Is All You Need 论文",
    "帮我复现一下 ResNet 那篇论文",
    "reproduce the BERT paper",
    "把 GPT-2 的官方实现复现一遍",
    "我想复现 transformer 那篇经典论文",
    "复现一篇关于扩散模型的论文",
    "reproduce ViT",
    "帮我看看这篇 paper 的代码能不能复现",
    "复现一下 LLaMA 的开源实现",
    "reproduction of the original Transformer",
    # ── 框架评测类 ──
    "对比 PyTorch 和 TensorFlow 的性能",
    "帮我比较 vLLM 和 llama.cpp 的推理速度",
    "benchmark 一下 HuggingFace transformers",
    "框架选型：LangChain vs LlamaIndex",
    "评测一下这两个推理框架",
    "compare TensorRT and ONNX Runtime",
    # ── 代码执行类 ──
    "帮我跑一段 Python 代码",
    "执行这个脚本",
    "run this script",
    "帮我把这个仓库跑起来",
    "运行一下这份代码看看结果",
    # ── 通用 / 边界类（设计上交给 LLM 分类）──
    "你好，你能做什么？",
    "今天天气怎么样？",
    "帮我写一封邮件",
    "帮我做个 PPT 大纲",
    "解释一下什么是残差连接",
    "分析 encoder-decoder 结构",
    "帮我分析这段代码为什么报错",
    "推荐几篇值得读的论文",
    "什么是 transformer",
]


def main() -> None:
    hits = 0
    for i, query in enumerate(SAMPLES, 1):
        route = route_by_rule(query)
        label = route.value if route else "— 交给 LLM"
        if route is not None:
            hits += 1
        print(f"{i:02d}. [{label:>22}] {query}")
    total = len(SAMPLES)
    print(f"\n规则快速路径命中 {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    main()
