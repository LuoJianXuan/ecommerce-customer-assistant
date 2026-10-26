"""Load the curated demo snapshot into an empty knowledge table."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description="向空知识库导入演示知识，不覆盖已有内容")
    parser.add_argument("--snapshot",type=Path,default=ROOT/"evals/releases/2026-10-26/knowledge_snapshot.json")
    parser.add_argument("--plan",action="store_true",help="只检查文件，不连接服务")
    args=parser.parse_args()
    rows=json.loads(args.snapshot.read_text(encoding="utf-8"))
    if not rows or any(r.get("governance_status")!="active" for r in rows):
        parser.error("只接受非空且全部 active 的公开知识快照")
    if len({r["id"] for r in rows})!=len(rows):
        parser.error("快照中知识编号重复")
    if args.plan:
        print(f"Validated active knowledge blocks: {len(rows)}; database/vectorization not executed.")
        return
    sys.path.insert(0,str(ROOT))
    from sqlalchemy import func,select
    from app.db import get_session_factory
    from app.knowledge.models import KnowledgeChunk
    from app.knowledge.vector_store import MilvusKnowledgeStore
    from app.knowledge.embeddings import BgeM3Embedder
    from app.knowledge.repository import vectorize_pending
    factory=get_session_factory()
    store=MilvusKnowledgeStore()
    with factory() as session:
        if session.scalar(select(func.count()).select_from(KnowledgeChunk)):
            parser.error("知识表非空，已停止。请使用专用空演示库；已有 pending 块可用 build --resume 补齐。")
        store.ensure_collection()
        if store.client.query(store.collection,filter="",output_fields=["chunk_id"],limit=1):
            parser.error("向量集合非空，已停止。请使用专用空集合，避免混入已有知识。")
        fields=["id","category","questions","answer","section_path","content_type","is_key_clause"]
        session.add_all(KnowledgeChunk(**{k:r[k] for k in fields},governance_status="active",vectorize_status="pending") for r in rows)
        session.commit()
    count=vectorize_pending(factory,BgeM3Embedder.get(),store)
    print(f"Loaded demo knowledge; vectorized {count} blocks.")

if __name__=="__main__":
    main()
