"""机制3：指标重算防伪（⭐ Sea Harness 灵魂）。

对应 DESIGN.md 2.4 节"机制 3"。
对应 Sea 的 benchmark_harness.go 从 predictions.jsonl 重算指标。

为什么需要：
    LLM 跑完 benchmark 后可能直接写一个 {"accuracy": 0.99} 到 metrics.json 谎报成功。
    本模块从 predictions.jsonl 逐样本重算指标，和 LLM 上报的不一致就判失败。
    这是 Harness 区别于"普通代码执行器"的核心。

实现说明：DESIGN 建议 sklearn，但为避免引入 40MB 依赖，这里用纯 Python
实现 accuracy / macro_f1 / mse / mae（数学定义一致，可交叉验证）。
"""

import json
from pathlib import Path


def recompute_metrics(predictions_path: str) -> dict:
    """从 predictions.jsonl 重算指标，作为 ground truth。

    predictions.jsonl 每行一个 JSON：{"y_true": ..., "y_pred": ...}
    空文件返回 {"error": "empty predictions"}。

    分类判定：y_true 全是 str，或唯一值 < 20 → accuracy + macro_f1；
    否则按回归 → mse + mae。
    """
    path = Path(predictions_path)
    if not path.exists():
        return {"error": "empty predictions"}

    preds = []
    with path.open(encoding="utf-8") as f:  # with open 防 fd 泄漏
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                return {"error": f"invalid json line: {line[:100]}"}

    if not preds:
        return {"error": "empty predictions"}

    y_true = [p["y_true"] for p in preds]
    y_pred = [p["y_pred"] for p in preds]

    is_classification = (
        all(isinstance(y, str) for y in y_true)
        or len(set(y_true)) < 20
    )
    if is_classification:
        return {
            "accuracy": _accuracy(y_true, y_pred),
            "macro_f1": _macro_f1(y_true, y_pred),
        }
    return {
        "mse": _mse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
    }


def _accuracy(y_true: list, y_pred: list) -> float:
    """准确率：预测正确的比例。"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _macro_f1(y_true: list, y_pred: list) -> float:
    """宏平均 F1：每个类各算 F1 再取平均（类不均衡时比 accuracy 更公平）。"""
    labels = sorted(set(y_true) | set(y_pred), key=str)
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def _mse(y_true: list, y_pred: list) -> float:
    """均方误差。"""
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def _mae(y_true: list, y_pred: list) -> float:
    """平均绝对误差。"""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def verify_metrics(
    reported: dict, recomputed: dict, tolerance: float = 1e-4
) -> tuple[bool, str]:
    """校验 LLM 上报的指标 vs 重算的指标。

    容差 1e-4 防浮点误差。返回 (一致, 原因)。不一致时原因含"疑似伪造"。
    注意：如果 recomputed 含 "error" key，调用方应直接判失败，不要调本函数。
    """
    for key in recomputed:
        if key == "error":
            continue
        if key not in reported:
            return False, f"LLM 上报缺少指标：{key}"
        try:
            diff = abs(float(reported[key]) - float(recomputed[key]))
        except (TypeError, ValueError):
            return False, f"指标 {key} 不是数值：reported={reported[key]!r}"
        if diff > tolerance:
            return False, (
                f"指标 {key} 不一致：LLM 上报 {reported[key]}, "
                f"重算 {recomputed[key]}（疑似伪造）"
            )
    return True, ""
