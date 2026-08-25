"""机制1：Patch Policy 静态校验（防 LLM 越权）。

对应 DESIGN.md 2.4 节"机制 1"。
对应 Sea 的 paper_debug_harness.go patch policy 静态拒绝。

为什么需要：
    LLM 修复代码时可能"偷懒"——补丁里加 subprocess.run("pip install xxx")
    绕过依赖恢复、加 mock_metric = 0.99 伪造指标、加 requests.get 突破断网。
    这些都要静态拦截。

调用时机：Harness 调 LLM 修复后、写入文件前，必须先过 validate_patch。
校验不过直接算修复失败，进入下一次重试。
"""

import re

# 禁止模式：LLM 修复代码时不得引入这些构造
# (正则, 拒绝原因)——原因会回喂给 LLM 让它换方案
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"pip\s+install", "禁止在补丁里装包，依赖恢复走 install_dependencies 节点"),
    (r"subprocess\.", "禁止 subprocess，沙箱断网会卡死"),
    (r"os\.system\s*\(", "禁止 os.system，用受控的 sandbox.exec_in 执行命令"),
    (r"mock.*metric|fake.*metric", "禁止伪造指标，metrics 由指标重算校验"),
    (r"verify\s*=\s*False", "禁止关闭 SSL 验证"),
    (r"shell\s*=\s*True", "禁止 shell=True，命令注入风险"),
    (r"requests\.(get|post)|httpx\.", "禁止外联，沙箱断网会失败"),
    (r"urllib\.request", "禁止 urllib 外联，沙箱断网会失败"),
    (r"socket\.socket\s*\(", "禁止裸 socket，沙箱断网会失败"),
    (r"eval\s*\(|exec\s*\(", "禁止 eval/exec，任意代码执行风险"),
    (r"__import__\s*\(", "禁止动态导入绕过静态校验"),
    (r"open\s*\(\s*['\"]/", "禁止读写workspace外的绝对路径文件"),
]


def validate_patch(code: str) -> tuple[bool, str]:
    """校验补丁是否引入禁止模式。

    返回 (合法, 原因)。合法为 True 时原因为空字符串。
    逐条正则扫描，命中即返回（首个命中的原因回喂 LLM）。
    """
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            return False, f"Patch Policy 拒绝：{reason}（命中模式: {pattern}）"
    return True, ""
