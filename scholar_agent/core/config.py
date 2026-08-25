"""配置加载：从环境变量 / .env 读取配置。

用 pydantic-settings 的 BaseSettings，自动从 .env 文件读值 + 类型转换 + 校验。
参考 career-agent-sim 的 config 风格：环境枚举 + 按环境套默认 + 生产强校验。

.env 和 config.py 的分工：
    .env 文件：存"值"（API key、端口、URL 等环境特定的敏感信息）
    config.py：定义"结构"（字段名、类型、默认值、校验、环境差异化配置）

用法：
    from scholar_agent.core.config import settings
    print(settings.LLM_API_KEY)
    print(settings.redis_url)  # property 自动拼接
"""

from enum import Enum
from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """运行环境枚举（str+Enum 双继承，可直接当字符串用）。

    - development：本地开发，DEBUG 开、日志彩色
    - test：跑测试，DEBUG 开、日志彩色
    - production：生产，DEBUG 关 + 强校验（密钥不达标直接启动失败）
    """

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


# APP_ENV 别名映射（dev / prod 都能识别）
_ENV_MAP = {
    "dev": "development",
    "develop": "development",
    "prod": "production",
    "production": "production",
    "test": "test",
}


class Settings(BaseSettings):
    """全局配置，通过环境变量或 .env 文件加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── 运行环境 ───
    APP_ENV: str = "development"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    PROJECT_NAME: str = "scholar-agent"
    API_PREFIX: str = "/api/v1"

    # ─── LLM（OpenAI 兼容，默认 DeepSeek）───
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4000
    LLM_MAX_RETRIES: int = 3

    # ─── Redis ───
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_MAX_CONNECTIONS: int = 20

    # ─── Docker 沙箱 ───
    SANDBOX_IMAGE: str = "python:3.12-slim"
    SANDBOX_MEM_LIMIT: str = "512m"
    SANDBOX_CPU_QUOTA: int = 100000
    SANDBOX_PIDS_LIMIT: int = 256
    SANDBOX_TIMEOUT: int = 300
    SANDBOX_WORKSPACE_ROOTS: str = ""  # 留空则用系统临时目录

    # ─── GitHub（搜 repo 用）───
    GITHUB_TOKEN: str = ""

    # ─── API ───
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> object:
        """兼容三种写法：JSON 数组 / 逗号分隔字符串 / 已是 list。

        不加这个 validator，pydantic-settings 会把 .env 里的
        CORS_ORIGINS=http://localhost:3000 当 JSON 解析而直接报错。
        """
        if isinstance(v, str):
            # 提前返回：JSON 数组格式（["a","b"]）直接解析
            if v.strip().startswith("["):
                import json

                return json.loads(v)
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # ─── 日志 ───
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | console

    # ─── LangSmith（可选，链路追踪）───
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "scholar-agent"

    @model_validator(mode="after")
    def _resolve_env_and_guard(self):
        """环境差异化配置 + 生产强校验。"""

        # 1) ENVIRONMENT 跟随 APP_ENV
        self.ENVIRONMENT = Environment(
            _ENV_MAP.get(self.APP_ENV.lower(), self.APP_ENV.lower())
        )

        # 2) 按环境套默认（环境变量优先级最高，已显式设置的不覆盖）
        env_defaults = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING",
            },
        }
        for key, val in env_defaults.get(self.ENVIRONMENT, {}).items():
            if key not in self.model_fields_set:
                setattr(self, key, val)

        # 3) 生产环境强校验（fail-fast）
        if self.ENVIRONMENT == Environment.PRODUCTION:
            assert self.DEBUG is False, "生产环境 DEBUG 必须为 False"
            assert self.CORS_ORIGINS != ["*"], "生产环境 CORS 不允许通配 '*'"
            assert self.LLM_API_KEY, "生产环境必须设置 LLM_API_KEY"
            assert self.GITHUB_TOKEN, "生产环境必须设置 GITHUB_TOKEN"

        return self

    @property
    def redis_url(self) -> str:
        """Redis 连接 URL，自动拼接 password。"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache
def get_settings() -> Settings:
    """全局单例（lru_cache 确保只实例化一次）。"""
    return Settings()


# 模块级单例，便于直接 import 使用
settings = get_settings()
