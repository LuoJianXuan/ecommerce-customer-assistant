"""按审计证据白名单恢复被误删的对话挖掘知识；默认只预览。"""

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session_factory
from app.knowledge.embeddings import BgeM3Embedder
from app.knowledge.models import KnowledgeChunk
from app.knowledge.repository import insert_chunks, vectorize_pending
from app.knowledge.splitter import SplitChunk
from app.knowledge.vector_store import MilvusKnowledgeStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("tmp/ch09-mined-refusal-cleanup.json"),
    )
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))["items"]
    by_id = {item["id"]: item for item in evidence}
    missing = set(args.ids) - by_id.keys()
    if missing:
        parser.error(f"审计证据中没有这些 ID: {sorted(missing)}")
    selected = [by_id[row_id] for row_id in args.ids]
    for item in selected:
        print(f"#{item['id']} {item['question']}\n  {item['answer']}")
    if not args.apply:
        print("dry-run：未写入 MySQL/Milvus")
        return 0

    factory = get_session_factory()
    chunks = []
    with factory() as session:
        for index, item in enumerate(selected):
            exists = session.scalar(
                select(KnowledgeChunk.id).where(
                    KnowledgeChunk.questions == item["question"],
                    KnowledgeChunk.answer == item["answer"],
                )
            )
            if exists is None:
                chunks.append(SplitChunk(
                    category="对话挖掘",
                    questions=item["question"],
                    answer=item["answer"],
                    section_path="对话挖掘>误删恢复",
                    content_type="faq",
                    is_key_clause=0,
                    source_order=index,
                ))
        row_ids = insert_chunks(session, chunks)
        session.commit()

    if row_ids:
        store = MilvusKnowledgeStore()
        store.ensure_collection()
        vectorize_pending(factory, BgeM3Embedder.get(), store, row_ids=row_ids)
    print(f"已恢复并向量化 {len(row_ids)} 条；已存在 {len(selected) - len(row_ids)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
