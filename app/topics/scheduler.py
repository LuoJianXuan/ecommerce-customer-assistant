"""ch10 自动攒批调度器；配置默认关闭，不影响手动批量归类。"""

import logging
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)
_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _run() -> None:
    from app.topics.api import get_topic_service

    settings = get_settings()
    interval = max(10.0, settings.topic_auto_batch_interval_minutes * 60)
    while not _stop_event.wait(interval):
        try:
            service = get_topic_service()
            pending = service.pending_count()
            if pending >= settings.topic_auto_batch_min_pending:
                result = service.classify_batch(
                    limit=settings.topic_auto_batch_limit, reclassify=False
                )
                logger.info("主题自动攒批完成:%s", result)
        except Exception:
            logger.exception("主题自动攒批失败")


def start_topic_scheduler() -> None:
    global _thread
    if not get_settings().topic_auto_batch_enabled or (_thread and _thread.is_alive()):
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_run, name="topic-auto-batch", daemon=True)
    _thread.start()


def stop_topic_scheduler() -> None:
    _stop_event.set()
