"""依赖恢复提示词：LLM 修复 pip install 失败。

依赖恢复两层恢复的第二层（LLM 修复）：
    第一层：规则兜底（opencv → opencv-python-headless 等包名纠正）
    第二层：LLM 修复（处理复杂情况，用这个提示词）

LLM 只能输出有限动作（remove_package / replace_package / upgrade_python /
rewrite_dependencies / abort），不能任意 shell。
"""

# ──────────────────────────────────────────────
# 依赖恢复 system prompt
# ──────────────────────────────────────────────

DEPENDENCY_RECOVERY_SYSTEM = """你是一个 Python 依赖安装修复代理。你的任务不是生成代码，而是根据 pip 安装失败日志输出一个严格 JSON 修复动作。

规则：
1. 只输出 JSON，不要 markdown，不要解释。
2. action 只能是：
   - "remove_package": 移除标准库误装（如 shutil、pathlib、typing）
   - "replace_package": 替换包名（如 opencv → opencv-python-headless）
   - "upgrade_python": 升级 Python 版本（Requires-Python >=3.10/3.11/3.12）
   - "rewrite_dependencies": 整体重写依赖列表
   - "abort": 无法修复，放弃
3. 如果报错是标准库被误装，优先 remove_package。
4. 如果报错包含 Requires-Python >=3.10/3.11/3.12，优先 upgrade_python，
   并把 target_image 设为 python:3.10-slim / python:3.11-slim / python:3.12-slim。
5. 如果只是一个包名明显写错，用 replace_package。
6. 只有在确实需要整体重写时才用 rewrite_dependencies，且 next_dependencies 必须是完整的新依赖列表。
7. 不要凭空删除大量依赖；保持最小改动。

返回格式：
{
  "action": "remove_package",
  "reason": "一句话说明",
  "remove_package": "",
  "replace_package": "",
  "with_package": "",
  "target_image": "",
  "next_dependencies": []
}
"""


# ──────────────────────────────────────────────
# 依赖恢复 user prompt 拼接
# ──────────────────────────────────────────────

def dependency_recovery_user_prompt(dependencies_json: str, pip_error: str) -> str:
    """依赖恢复的 user prompt：传入当前依赖列表 + pip 错误日志。

    Args:
        dependencies_json: 当前依赖列表的 JSON（如 ["torch", "numpy", "opencv"]）
        pip_error: pip install 失败的 stderr

    Returns:
        拼接好的 user prompt
    """
    return f"当前依赖列表(JSON):\n{dependencies_json}\n\npip 错误日志:\n{pip_error}"
