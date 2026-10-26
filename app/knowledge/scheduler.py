"""挖知识定时任务（APScheduler 内嵌 FastAPI，间隔 .env 可配，<=0 关闭）。

CLI 手动入口与定时任务共用 run_mining（app.knowledge.mine）。
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import get_settings

logger = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def start_mining_scheduler() -> AsyncIOScheduler | None:
    """在 FastAPI lifespan 中调用；间隔<=0 时不启动（返回 None）。"""
    global _scheduler
    settings = get_settings()
    if settings.mine_interval_hours <= 0:
        logger.info("挖知识定时任务已关闭（MINE_INTERVAL_HOURS<=0）")
        return None
    if _scheduler is not None:
        return _scheduler

    from app.knowledge.mine import run_mining

    _scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

    async def _job() -> None:
        try:
            # 挖矿内含同步重活（embed/Milvus），整个流程放独立线程+独立事件循环，
            # 不阻塞 FastAPI 主事件循环（在线 SSE 不受挖掘影响）
            import asyncio

            stats = await asyncio.to_thread(asyncio.run, run_mining())
            logger.info("挖知识定时任务完成: %s", stats)
        except Exception:
            logger.exception("挖知识定时任务失败")

    _scheduler.add_job(
        _job,
        "interval",
        hours=settings.mine_interval_hours,
        id="knowledge-mining",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("挖知识定时任务已启动，间隔 %s 小时", settings.mine_interval_hours)
    return _scheduler


def stop_mining_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
