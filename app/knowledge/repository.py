"""知识库 MySQL 仓储：chunk 落库、prev/next 两阶段回填、双写状态机。

幂等设计（计划 v4 + 修正）：
- vector_id 从 chunk_id 确定性生成（"kb-{id}"）——重跑 upsert 同主键覆盖，
  不产生孤儿向量（比随机 uuid 真正幂等，修正记录见 dev-notes/ch03.md）；
- --resume 语义 = vectorize_pending 只扫 pending；
- 同 source 重复导入 = replace_source 整体替换（删旧行+旧向量，重插）。
"""

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.models import KnowledgeChunk
from app.knowledge.splitter import SplitChunk


def insert_chunks(session: Session, chunks: list[SplitChunk]) -> list[int]:
    """批量落库（pending）+ 两阶段回填 prev/next（自引用 FK 不能插入时互指）。"""
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda c: c.source_order)
    rows = [
        KnowledgeChunk(
            category=c.category,
            questions=c.questions,
            answer=c.answer,
            section_path=c.section_path,
            content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        for c in ordered
    ]
    session.add_all(rows)
    session.flush()  # 拿自增 id
    for i in range(len(rows) - 1):
        rows[i].next_chunk_id = rows[i + 1].id
        rows[i + 1].prev_chunk_id = rows[i].id
    session.flush()
    return [r.id for r in rows]


def replace_source(
    session: Session, source: str, chunks: list[SplitChunk], store=None
) -> list[int]:
    """同一来源整体替换：删旧行；store 给定时连带删除旧向量，再重插。"""
    old_ids = session.scalars(
        select(KnowledgeChunk.id).where(
            KnowledgeChunk.section_path.like(f"{source}>%")
            | (KnowledgeChunk.category == source)
        )
    ).all()
    if old_ids:
        session.query(KnowledgeChunk).where(
            KnowledgeChunk.id.in_(old_ids)
        ).delete(synchronize_session=False)
        if store is not None:
            store.delete_by_chunk_ids(old_ids)
    session.flush()
    return insert_chunks(session, chunks)


def vectorize_pending(
    session_factory: sessionmaker[Session],
    embedder,
    store,
    limit: int | None = None,
    row_ids: list[int] | None = None,
) -> int:
    """双写状态机：pending → embed → Milvus upsert → 回填 vector_id+done。

    逐块提交：中断时已完成的块保持 done，剩余块留 pending 供重跑补齐。
    返回本次完成数。任何块的失败立即抛出（调用方决定重跑时机）。
    """
    from app.knowledge.embeddings import build_vector_text

    with session_factory() as session:
        stmt = select(KnowledgeChunk).where(
            KnowledgeChunk.vectorize_status == "pending",
            KnowledgeChunk.governance_status == "active",
        )
        if row_ids is not None:
            stmt = stmt.where(KnowledgeChunk.id.in_(row_ids))
        if limit:
            stmt = stmt.limit(limit)
        rows = session.scalars(stmt).all()
        ids = [r.id for r in rows]

    done = 0
    for row_id in ids:
        vector_id = f"kb-{row_id}"
        with session_factory() as session:
            row = session.get(KnowledgeChunk, row_id)
            if (
                row is None
                or row.vectorize_status == "done"
                or row.governance_status != "active"
            ):
                continue  # 并发下已被处理
            text = build_vector_text(row.category, row.questions, row.answer)
            category = row.category
        vec = embedder.embed_texts([text])[0]
        store.upsert(vector_id, row_id, text, category, vec)  # 失败即抛：本块留在 pending
        with session_factory() as session:
            row = session.get(KnowledgeChunk, row_id)
            row.vector_id = vector_id
            row.vectorize_status = "done"
            session.commit()
        done += 1
    return done
