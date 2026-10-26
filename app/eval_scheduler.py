"""ch09 每日评估调度。"""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import get_settings
from app.eval_run_lock import claim_eval_run, release_eval_run

logger = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None


def start_eval_scheduler() -> AsyncIOScheduler | None:
    global _scheduler
    settings = get_settings()
    if settings.eval_daily_hour <= 0:
        logger.info("每日评估已关闭（EVAL_DAILY_HOUR<=0）")
        return None
    if _scheduler is not None:
        return _scheduler

    from evals.ch09.run_pipeline import run_pipeline

    async def _job() -> None:
        if not claim_eval_run():
            logger.info("已有评估在运行，本次每日评估跳过")
            return
        try:
            run_id = await asyncio.to_thread(asyncio.run, run_pipeline("scheduled"))
            logger.info("每日评估完成 run_id=%s", run_id)
        except Exception:
            logger.exception("每日评估失败")
        finally:
            release_eval_run()

    _scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
    _scheduler.add_job(
        _job,
        "cron",
        hour=settings.eval_daily_hour,
        id="daily-evaluation",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    return _scheduler


def stop_eval_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
