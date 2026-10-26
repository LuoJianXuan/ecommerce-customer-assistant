"""幂等应用 ch09 数据飞轮表结构。"""

import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_engine


def main() -> None:
    engine = get_engine()
    tables = set(inspect(engine).get_table_names())
    ddl = {
        "review_queue": """
            CREATE TABLE review_queue (
              id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
              std_question TEXT NOT NULL,
              sample_answer TEXT NOT NULL,
              source VARCHAR(32) NOT NULL DEFAULT '数据飞轮',
              occurrence_count INT UNSIGNED NOT NULL DEFAULT 1,
              status ENUM('待审','已通过','已驳回') NOT NULL DEFAULT '待审',
              approved_answer TEXT NULL,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (id), KEY idx_review_status (status), KEY idx_review_source (source)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        "retrieval_snapshots": """
            CREATE TABLE retrieval_snapshots (
              id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
              conversation_id BIGINT UNSIGNED NOT NULL,
              query TEXT NOT NULL, chunks JSON NOT NULL,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id), KEY idx_snapshot_conv_time (conversation_id, created_at),
              CONSTRAINT fk_snapshot_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        "eval_runs": """
            CREATE TABLE eval_runs (
              id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
              `trigger` ENUM('manual','scheduled') NOT NULL,
              case_count INT UNSIGNED NOT NULL, metrics JSON NOT NULL,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id), KEY idx_eval_run_time (created_at),
              KEY idx_eval_trigger (`trigger`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    }
    with engine.begin() as conn:
        for table, statement in ddl.items():
            if table not in tables:
                conn.execute(text(statement))
        review_columns = {c["name"] for c in inspect(engine).get_columns("review_queue")}
        if "source" not in review_columns:
            conn.execute(text(
                "ALTER TABLE review_queue "
                "ADD COLUMN source VARCHAR(32) NOT NULL DEFAULT '数据飞轮' AFTER sample_answer, "
                "ADD KEY idx_review_source (source)"
            ))
        columns = {c["name"] for c in inspect(engine).get_columns("low_confidence_questions")}
        if "retrieval_snapshot" not in columns:
            conn.execute(text("ALTER TABLE low_confidence_questions ADD COLUMN retrieval_snapshot JSON NULL"))
        if "matched_review_id" not in columns:
            conn.execute(text(
                "ALTER TABLE low_confidence_questions "
                "ADD COLUMN matched_review_id BIGINT UNSIGNED NULL, "
                "ADD KEY idx_lcq_review (matched_review_id), "
                "ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id)"
            ))
    print("ch09 schema ready")


if __name__ == "__main__":
    main()
