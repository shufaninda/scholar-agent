"""Redis 短期记忆：LPUSH/LTRIM 滑动窗口保留最近 N 轮对话。

对应 Sea 的 IntentMemoryStore（memory_redis.go）。
对应 DESIGN.md 1 节"Redis 记忆"。

为什么需要：
    意图识别不能只看当前这一句——用户可能说"那换个框架对比"，
    如果不知道上一句是"复现 Attention 论文"，就无法理解"换个框架"指什么。
    Redis 短期记忆加载最近 N 轮对话作为上下文，让 LLM 理解多轮对话。

Redis 数据结构：
    Key:   sa:intent:turns:{session_id}  (LIST)
    Value: JSON 序列化的 StoredTurn
    操作:
        - LPUSH: 头插新对话（新的在前面）
        - LTRIM: 保留最近 30 轮（滑动窗口）
        - LRANGE: 读最近 N 轮（LPUSH 是新→旧，读出后要反转成时间正序）
        - EXPIRE: TTL 7 天

降级语义：
    Redis 不可用 → NoopMemoryStore（全部 no-op，不阻断主流程）
    Redis 读失败 → 用空列表继续 LLM（不阻断）
"""

import json
import time
from typing import Any

from redis.asyncio import Redis


class IntentMemoryStore:
    """Redis 短期记忆：加载最近 N 轮对话补充上下文。

    对应 Sea 的 IntentMemoryStore（memory_redis.go）。

    用法：
        store = IntentMemoryStore(redis)
        memory = await store.load_recent_turns(session_id, limit=10)
        # memory 是 list[dict]，时间正序（旧→新）
        ...  # 把 memory 拼进 LLM prompt
        await store.append_turn(session_id, "user", "复现 Attention 论文")
    """

    KEY_PREFIX = "sa:intent:turns:"
    MAX_TURNS = 30          # LTRIM 保留最近 30 轮
    DEFAULT_FETCH = 10      # 默认读最近 10 轮
    TTL_SECONDS = 7 * 24 * 3600  # 7 天

    def __init__(self, redis: Redis | None = None):
        """初始化。redis=None 时降级为 Noop（全部 no-op）。"""
        self.redis = redis

    async def load_recent_turns(
        self, session_id: str, limit: int = DEFAULT_FETCH
    ) -> list[dict[str, Any]]:
        """加载最近 N 轮对话（时间正序：旧→新）。

        Redis LPUSH 是新→旧，LRANGE 读出后需要反转成时间正序。

        降级：Redis 不可用或读失败 → 返回空列表（不阻断主流程）。
        """
        if self.redis is None:
            return []

        try:
            key = f"{self.KEY_PREFIX}{session_id}"
            # LRANGE 0 limit-1：读前 limit 个（新的在前面）
            raw_turns = await self.redis.lrange(key, 0, limit - 1)
            # 反转成时间正序（旧→新）
            turns = []
            for raw in reversed(raw_turns):
                if isinstance(raw, bytes):
                    raw = raw.decode()
                turns.append(json.loads(raw))
            return turns
        except Exception:
            # Redis 读失败不阻断，用空列表继续
            return []

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent_type: str | None = None,
    ) -> None:
        """追加一轮对话到记忆（LPUSH + LTRIM 滑动窗口 + EXPIRE）。

        降级：Redis 不可用或写失败 → 静默忽略（不阻断主流程）。
        """
        if self.redis is None:
            return

        try:
            key = f"{self.KEY_PREFIX}{session_id}"
            turn = {
                "role": role,
                "content": content,
                "ts_ms": int(time.time() * 1000),
                "intent_type": intent_type,
            }
            # Pipeline 一次性执行 LPUSH + LTRIM + EXPIRE
            pipe = self.redis.pipeline()
            pipe.lpush(key, json.dumps(turn))           # 头插新的
            pipe.ltrim(key, 0, self.MAX_TURNS - 1)      # 保留最近 30 轮
            pipe.expire(key, self.TTL_SECONDS)          # 刷新 TTL
            await pipe.execute()
        except Exception:
            # Redis 写失败不阻断，静默忽略
            pass
