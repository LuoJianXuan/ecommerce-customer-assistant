"""ch07 文件日志：log/app.log（RotatingFileHandler 5MB×3）。

model_ctx / history_ctx / summary / context 等日志器统一落这里（grep 验收目标）；
uvicorn 自身日志走各自 handler，不进本文件。
"""

import logging
import logging.handlers
from pathlib import Path


def setup_file_logging(log_dir: str = "log") -> None:
    path = Path(log_dir)
    path.mkdir(exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return  # 幂等（lifespan 可能被多 app 实例触发）
    handler = logging.handlers.RotatingFileHandler(
        path / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    # root 默认 WARNING：不抬到 INFO 的话 model_ctx/history_ctx/summary 的 INFO
    # 在 logger 层就被过滤，根本到不了文件 handler（实测翻车）
    if root.level == logging.WARNING or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
