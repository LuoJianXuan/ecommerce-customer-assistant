"""清点并可选清理历史“拒答型”对话挖掘知识；默认只导出证据，不删除。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session_factory
from app.knowledge.mine import is_rejection_answer
from app.knowledge.models import KnowledgeChunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="确认双删 MySQL 与 Milvus；默认 dry-run")
    parser.add_argument("--ids", nargs="+", type=int, help="人工确认要删除的知识块 ID")
    parser.add_argument("--output", type=Path, default=Path("tmp/ch09-mined-refusal-cleanup.json"))
    args = parser.parse_args()
    if args.apply and not args.ids:
        parser.error("--apply 必须同时提供人工确认的 --ids")

    factory = get_session_factory()
    with factory() as session:
        rows = session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.category == "对话挖掘")
        ).all()
        if args.ids:
            selected = set(args.ids)
            candidates = [row for row in rows if row.id in selected]
            missing = selected - {row.id for row in candidates}
            if missing:
                parser.error(f"不是对话挖掘知识块或不存在: {sorted(missing)}")
        else:
            candidates = [row for row in rows if is_rejection_answer(row.answer)]
        evidence = [{
            "id": row.id,
            "question": row.questions,
            "answer": row.answer,
            "vector_id": row.vector_id,
        } for row in candidates]

    args.output.write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "applied": args.apply,
        "items": evidence,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选 {len(evidence)} 条；证据已写 {args.output}")
    if not args.apply or not candidates:
        return 0

    from app.knowledge.vector_store import MilvusKnowledgeStore

    ids = [row.id for row in candidates]
    store = MilvusKnowledgeStore()
    store.delete_by_chunk_ids(ids)
    with factory() as session:
        session.query(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)).delete(
            synchronize_session=False
        )
        session.commit()
    print(f"已从 Milvus 与 MySQL 删除 {len(ids)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
