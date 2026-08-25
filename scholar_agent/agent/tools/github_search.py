"""GitHub 搜索工具：用 PyGithub 搜论文的复现 repo。

纯 API 调用，不调 LLM——LLM 在意图识别阶段已抽取搜索关键词（PaperSearchFields）。

质量过滤：
    - stargazers_count > 10（过滤玩具 repo）
    - 最近一年内有 push（过滤弃坑 repo）
"""

import logging

from github import Github

from scholar_agent.core.config import settings

logger = logging.getLogger(__name__)

MIN_STARS = 10
MAX_CANDIDATES = 5


def search_repos(
    query: str,
    token: str | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> list[dict]:
    """搜 GitHub repo，返回按 star 降序的候选列表。

    返回：[{"full_name", "html_url", "stars", "pushed_at"}]
    异常向上抛（调用方 ResearchCodingAgent 决定降级）。
    """
    gh = Github(token or settings.GITHUB_TOKEN or None)
    results = gh.search_repositories(query, sort="stars", order="desc")
    return [
        {
            "full_name": repo.full_name,
            "html_url": repo.html_url,
            "stars": repo.stargazers_count,
            "pushed_at": str(repo.pushed_at),
        }
        for repo in results[:max_candidates]
    ]


def pick_best_repo(candidates: list[dict], min_stars: int = MIN_STARS) -> str | None:
    """从候选里挑最优：star 达标 + 近一年活跃。无合格者返回 None。

    纯函数（不联网），方便单测质量过滤逻辑。
    """
    from datetime import datetime, timedelta, timezone

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    for repo in candidates:
        if repo["stars"] < min_stars:
            continue  # 提前跳过低星 repo
        pushed = datetime.fromisoformat(repo["pushed_at"])
        if pushed.tzinfo is None:
            pushed = pushed.replace(tzinfo=timezone.utc)
        if pushed < one_year_ago:
            continue  # 弃坑 repo 不要
        return repo["html_url"]
    return None
