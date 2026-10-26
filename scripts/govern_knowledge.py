"""历史知识治理：盘点、可恢复隔离、恢复。

默认只增加治理元数据字段并生成 JSON 盘点报告，不改变知识状态。--apply 会把历史自动挖掘和
误入客服库的业务无关材料标记为 quarantined，并从 Milvus 删除对应向量；MySQL 原文始终保留。
恢复时使用 --restore-ids 并指定审核后的正式类目，随后运行 build --resume。
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, select, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_engine, get_session_factory
from app.knowledge.models import KnowledgeChunk


UNRELATED_CATEGORIES = {"未分类"}  # 其他业务无关分类由命令行显式指定


def ensure_schema() -> None:
    engine = get_engine()
    columns = {c["name"] for c in inspect(engine).get_columns("knowledge_chunks")}
    statements = []
    if "governance_status" not in columns:
        statements.append(
            "ADD COLUMN governance_status "
            "ENUM('active','quarantined','pending_review') "
            "NOT NULL DEFAULT 'active' AFTER vectorize_status"
        )
    if "governance_reason" not in columns:
        statements.append(
            "ADD COLUMN governance_reason VARCHAR(255) NULL AFTER governance_status"
        )
    if "governed_at" not in columns:
        statements.append("ADD COLUMN governed_at DATETIME NULL AFTER governance_reason")
    if statements:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE knowledge_chunks " + ", ".join(statements)))
    indexes = {i["name"] for i in inspect(engine).get_indexes("knowledge_chunks")}
    if "idx_governance_status" not in indexes:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE knowledge_chunks "
                "ADD KEY idx_governance_status (governance_status)"
            ))


def reason_for(row) -> str | None:
    if row.category == "对话挖掘":
        return "legacy_conversation_mining_requires_review"
    if row.category in UNRELATED_CATEGORIES:
        return "unrelated_material_requires_review"
    return None


def inventory(factory) -> tuple[list, list[dict]]:
    with factory() as session:
        rows = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
    candidates = [row for row in rows if reason_for(row)]
    items = [
        {
            "id": row.id,
            "category": row.category,
            "question": row.questions,
            "reason": reason_for(row),
            "governance_status": getattr(row, "governance_status", "active"),
            "vector_id": row.vector_id,
        }
        for row in candidates
    ]
    return rows, items


def write_report(path: Path, rows: list, candidates: list[dict], applied: bool) -> None:
    category_counts = Counter(row.category for row in rows)
    reason_counts = Counter(item["reason"] for item in candidates)
    status_counts = Counter(getattr(row, "governance_status", "active") for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "applied": applied,
        "total_chunks": len(rows),
        "candidate_count": len(candidates),
        "status_counts": dict(status_counts),
        "category_counts": dict(category_counts.most_common()),
        "reason_counts": dict(reason_counts),
        "items": candidates,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def quarantine(factory, candidates: list[dict]) -> int:
    actionable = [item for item in candidates if item["governance_status"] != "quarantined"]
    ids = [item["id"] for item in actionable]
    if not ids:
        return 0
    reasons = {item["id"]: item["reason"] for item in actionable}
    now = datetime.now()
    with factory() as session:
        rows = session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))).all()
        for row in rows:
            row.governance_status = "quarantined"
            row.governance_reason = reasons[row.id]
            row.governed_at = now
            row.vectorize_status = "pending"
            row.vector_id = None
        session.commit()

    from app.knowledge.vector_store import MilvusKnowledgeStore

    MilvusKnowledgeStore().delete_by_chunk_ids(ids)
    return len(ids)


def restore(factory, ids: list[int], category: str) -> None:
    now = datetime.now()
    with factory() as session:
        rows = session.scalars(select(KnowledgeChunk).where(
            KnowledgeChunk.id.in_(ids),
            KnowledgeChunk.governance_status == "quarantined",
        )).all()
        found = {row.id for row in rows}
        missing = set(ids) - found
        if missing:
            raise ValueError(f"不是已隔离知识块或不存在: {sorted(missing)}")
        for row in rows:
            row.governance_status = "active"
            row.governance_reason = "restored_after_manual_review"
            row.governed_at = now
            row.category = category
            row.vectorize_status = "pending"
            row.vector_id = None
        session.commit()
    print("恢复状态已写入；运行 python -m app.knowledge.build --resume 重新向量化。")


def main() -> int:
    parser = argparse.ArgumentParser(description="客服知识库治理")
    parser.add_argument("--unrelated-category", action="append", default=[], help="额外隔离分类，可重复传入")
    parser.add_argument("--apply", action="store_true", help="应用隔离；默认只盘点")
    parser.add_argument("--restore-ids", nargs="+", type=int, help="恢复人工确认的隔离条目")
    parser.add_argument("--restore-category", help="恢复条目审核后的正式业务类目")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/knowledge-governance-inventory.json"),
    )
    args = parser.parse_args()
    UNRELATED_CATEGORIES.update(args.unrelated_category)
    if args.restore_ids and not args.apply:
        parser.error("--restore-ids 必须与 --apply 同时使用")
    if args.restore_ids and not args.restore_category:
        parser.error("--restore-ids 必须同时提供 --restore-category")
    if args.restore_category and not args.restore_ids:
        parser.error("--restore-category 只能与 --restore-ids 同时使用")

    ensure_schema()
    factory = get_session_factory()
    if args.restore_ids:
        restore(factory, args.restore_ids, args.restore_category)
        return 0

    rows, candidates = inventory(factory)
    write_report(args.output, rows, candidates, applied=False)
    print(f"总知识 {len(rows)}；治理候选 {len(candidates)}；报告 {args.output}")
    if not args.apply:
        return 0
    changed = quarantine(factory, candidates)
    rows, candidates = inventory(factory)
    write_report(args.output, rows, candidates, applied=True)
    print(f"本次新增隔离并从 Milvus 移除 {changed} 条；MySQL 原文保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
