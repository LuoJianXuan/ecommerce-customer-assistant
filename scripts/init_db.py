"""Initialize a local demo database; never drops tables or resets existing data."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ["schema.sql", "schema_ch03.sql", "schema_ch04.sql", "schema_ch06.sql",
         "schema_ch07.sql", "schema_ch08.sql"]

def prepare_sql(text: str) -> str:
    # ch07's column migration is applied individually after checking existing columns.
    text = re.sub(r"ALTER TABLE conversations\s+ADD COLUMN summary_upto_msg_id.*?;", "", text, flags=re.S)
    return re.sub(r"\bCREATE TABLE(?!\s+IF\s+NOT\s+EXISTS)\s+", "CREATE TABLE IF NOT EXISTS ", text, flags=re.I)

def execute_file(cursor, path: Path) -> None:
    cursor.execute(prepare_sql(path.read_text(encoding="utf-8-sig")))
    while cursor.nextset():
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="创建演示库表，不删除已有表或数据")
    parser.add_argument("--plan", action="store_true", help="只显示执行顺序，不连接数据库")
    parser.add_argument("--seed-demo-faq", action="store_true", help="仅在 FAQ 表为空时导入演示 FAQ")
    args = parser.parse_args()
    if args.plan:
        for name in FILES:
            print(f"sql/{name}")
        print("conversations: add missing summary anchors; apply_ch09_schema; apply_ch10_schema")
        if args.seed_demo_faq:
            print("sql/seed_faq.sql (only when faq is empty)")
        return

    sys.path.insert(0, str(ROOT))
    import pymysql
    from pymysql.constants import CLIENT
    from app.config import get_settings
    settings = get_settings()
    if not settings.db_password:
        parser.error("请先在本地 .env 中填写 DB_PASSWORD")
    connection = pymysql.connect(host=settings.db_host, port=settings.db_port,
        user=settings.db_user, password=settings.db_password, database=settings.db_name,
        charset="utf8mb4", client_flag=CLIENT.MULTI_STATEMENTS)
    try:
        with connection.cursor() as cursor:
            for name in FILES:
                execute_file(cursor, ROOT/"sql"/name)
                print(f"ready: {name}")
            cursor.execute("SHOW COLUMNS FROM conversations")
            existing = {r[0] for r in cursor.fetchall()}
            for name in ["summary_upto_msg_id", "layer1_from_msg_id"]:
                if name not in existing:
                    cursor.execute(f"ALTER TABLE conversations ADD COLUMN {name} BIGINT UNSIGNED NULL")
            if args.seed_demo_faq:
                cursor.execute("SELECT COUNT(*) FROM faq")
                if cursor.fetchone()[0] == 0:
                    execute_file(cursor, ROOT/"sql/seed_faq.sql")
                    print("seeded: demo FAQ")
                else:
                    print("kept: existing FAQ rows")
        connection.commit()
    finally:
        connection.close()

    from scripts.apply_ch09_schema import main as apply_ch09
    from scripts.apply_ch10_schema import main as apply_ch10
    apply_ch09()
    apply_ch10()
    print("Database initialization complete. No existing data was reset.")

if __name__ == "__main__":
    main()
