"""幂等应用 ch10 主题分类表结构。"""

import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_engine


DDL = """
CREATE TABLE topic_classifications (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question_id BIGINT UNSIGNED NOT NULL,
  labels JSON NOT NULL,
  scores JSON NOT NULL,
  model_version VARCHAR(128) NOT NULL,
  batch_id VARCHAR(64) NOT NULL,
  classified_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_topic_question (question_id),
  KEY idx_topic_batch (batch_id),
  KEY idx_topic_model (model_version),
  KEY idx_topic_time (classified_at),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id)
    REFERENCES low_confidence_questions (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def main() -> None:
    engine = get_engine()
    tables = set(inspect(engine).get_table_names())
    if "low_confidence_questions" not in tables:
        raise SystemExit("缺少 low_confidence_questions,请先运行 scripts/init_db.py 或应用 ch04、ch09 schema")
    if "topic_classifications" not in tables:
        with engine.begin() as conn:
            conn.execute(text(DDL))
        action = "created"
    else:
        action = "already-ready"
    columns = {c["name"] for c in inspect(engine).get_columns("topic_classifications")}
    expected = {
        "id", "question_id", "labels", "scores", "model_version", "batch_id", "classified_at"
    }
    missing = sorted(expected - columns)
    if missing:
        raise SystemExit(f"topic_classifications 结构不完整,缺列:{missing}")
    print(f"ch10 schema ready ({action})")


if __name__ == "__main__":
    main()
