"""手动与定时评估共用的进程内互斥。"""

import threading


_lock = threading.Lock()


def claim_eval_run() -> bool:
    return _lock.acquire(blocking=False)


def release_eval_run() -> None:
    _lock.release()
