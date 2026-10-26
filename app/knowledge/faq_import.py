"""FAQ 表导入 knowledge_chunks：questions=真实问法（ch03 需求），content_type=faq。

faq 表是 ch02 的 LIKE 数据源；导入后同样走双写向量化。
category 取 faq.category（售后/物流/会员/支付……）。
"""

from app.knowledge.repository import replace_source, vectorize_pending
from app.knowledge.splitter import SplitChunk
from app.models import Faq

SOURCE = "FAQ知识库"


def import_faq() -> int:
    from sqlalchemy import select

    from app.db import get_session_factory

    factory = get_session_factory()
    store = None
    with factory() as session:
        rows = session.scalars(select(Faq)).all()
        if not rows:
            print("[faq_import] faq 表为空，跳过")
            return 0
        chunks = [
            SplitChunk(
                category=row.category,
                questions=row.question,      # 真实问法
                answer=row.answer,
                section_path=f"{SOURCE}>{row.category}>{row.question[:40]}",
                content_type="faq",
                is_key_clause=0,
                source_order=i,
            )
            for i, row in enumerate(rows)
        ]
        from app.knowledge.vector_store import MilvusKnowledgeStore

        store = MilvusKnowledgeStore()
        store.ensure_collection()
        replace_source(session, SOURCE, chunks, store=store)
        session.commit()
        print(f"[faq_import] {len(chunks)} 条 FAQ 落库(pending)")

    from app.knowledge.embeddings import BgeM3Embedder

    done = vectorize_pending(factory, BgeM3Embedder.get(), store)
    print(f"[faq_import] 向量化完成 {done} 条")
    return done
