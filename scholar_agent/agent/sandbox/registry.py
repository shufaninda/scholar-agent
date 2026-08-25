"""container_id Redis 注册表：plan_id ↔ container_id 映射。

对应 DESIGN.md 2.2.2 节"container_id 持久化"。

为什么需要：
    Sea 的 mountPaths map[string]string 是内存映射，重启即丢，孤儿容器泄漏。
    Python 用 Redis 存 sandbox:{plan_id} → container_id 是真正的改进。
    Redis 已在技术栈里，不引入新依赖。

为什么不用数据库：
    容器 ID 是临时运行时状态，Redis TTL 自动兜底即可。

为什么不用内存 dict：
    重启即丢，无法跨进程共享，SSE 查不到运行中的容器。

【AI 生成】SandboxRegistry 类（你审）
"""

from redis.asyncio import Redis


class SandboxRegistry:
    """容器 ID 注册表：Redis 存 plan_id ↔ container_id 映射。

    redis=None 时全方法 no-op（Redis 不可用的降级态：
    索引丢失只影响"外部排查孤儿容器"，不影响执行主流程）。
    """

    KEY_PREFIX = "sandbox:"
    DEFAULT_TTL = 3600  # 1 小时，超时自动清索引（容器可能已被外部清理）

    def __init__(self, redis: Redis | None):
        self.redis = redis

    async def register(self, plan_id: str, container_id: str, ttl: int = DEFAULT_TTL) -> None:
        """注册容器 ID。"""
        if self.redis is None:
            return
        await self.redis.set(f"{self.KEY_PREFIX}{plan_id}", container_id, ex=ttl)

    async def get(self, plan_id: str) -> str | None:
        """获取容器 ID（bytes/str 统一转 str）。"""
        if self.redis is None:
            return None
        raw = await self.redis.get(f"{self.KEY_PREFIX}{plan_id}")
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    async def remove(self, plan_id: str) -> None:
        """删除索引（容器清理后调用）。"""
        if self.redis is None:
            return
        await self.redis.delete(f"{self.KEY_PREFIX}{plan_id}")
