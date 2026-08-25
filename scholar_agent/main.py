"""FastAPI 应用入口：路由注册 + 中间件 + 生命周期。

对应 Sea 的 cmd/server/main.go。
lifespan 职责：启动时组装 AppState（LLM/沙箱/Redis 都是惰性连接，
导入和启动不会因缺 Docker/Redis 而失败，真正用到时才连）。
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scholar_agent.agent.llm_client import LLMClient
from scholar_agent.agent.sandbox.docker_executor import DockerSandbox, SandboxRegistry
from scholar_agent.api.app_state import AppState, build_app_state
from scholar_agent.api.routes import router
from scholar_agent.core.config import settings
from scholar_agent.core.logging import logger


async def _build_state() -> AppState:
    """组装真实依赖的 AppState（惰性连接：这里只建对象不联网）。"""
    llm = LLMClient()
    redis = await _build_redis()
    sandbox = DockerSandbox(SandboxRegistry(redis))
    checkpointer = await _build_checkpointer(redis)
    return build_app_state(llm, sandbox, redis, checkpointer)


async def _build_redis():
    """Redis 连接（from_url 是惰性的，这里 ping 一次真探活；失败降级 None 走 Noop）。"""
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception as exc:  # noqa: BLE001 Redis 不可用不阻断启动
        logger.warning("redis_init_failed fallback=noop error=%s", exc)
        return None


async def _build_checkpointer(redis):
    """Checkpointer 装配：Redis 活着 → AsyncRedisSaver（真持久化，重启
    不丢，审批挂起可跨进程恢复）；任何失败 → InMemorySaver 兜底
    （进程内可用，重启丢，但 /runs 不断服务）。

    ⭐ interrupt() 强依赖 checkpointer（挂起点无处可写直接 RuntimeError），
    所以必须保证任何降级场景都有一个"口袋"——降级哲学的延续：
    Redis 挂了不是 /runs 不可用，只是失去跨重启持久化。
    """
    if redis is not None:
        try:
            # 延迟导入：包未安装（ImportError）与其他故障统一走兜底分支
            from langgraph.checkpoint.redis import AsyncRedisSaver

            # 用 URL 让 saver 自建连接：它对 decode_responses 等序列化
            # 参数有自己的要求，复用业务连接可能踩坑
            saver = AsyncRedisSaver(redis_url=settings.redis_url)
            await saver.setup()  # 建索引（幂等，重启安全）
            logger.info("checkpointer=redis")
            return saver
        except Exception as exc:  # noqa: BLE001 RedisSaver 任何故障都兜底
            logger.warning("redis_checkpointer_failed fallback=memory error=%s", exc)
    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("checkpointer=memory")
    return InMemorySaver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动组装 AppState，关闭清后台任务。"""
    logger.info("app_starting", host=settings.API_HOST, port=settings.API_PORT)
    app.state.scholar = await _build_state()
    yield
    # 等后台执行任务收尾（最多 5 秒，超了就不管——进程要退了）
    if app.state.scholar.tasks:
        await asyncio.wait(set(app.state.scholar.tasks), timeout=5)
    logger.info("app_stopped")


app = FastAPI(
    title="ScholarAgent",
    description="论文复现 + 框架评测 + 代码执行的 Agent 系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "version": "0.1.0"}
