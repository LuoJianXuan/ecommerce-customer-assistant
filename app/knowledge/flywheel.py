"""ch09 数据飞轮：检索快照、问题落池与待审归并。"""

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.models import LowConfidenceQuestion, RetrievalSnapshot


_merge_lock = asyncio.Lock()

NORMALIZE_SYSTEM = """你是客服知识缺口整理员。把用户口语原话改写成一条标准 FAQ 问题，
并给一条供审核人参考的示例答案。只输出 JSON：
{{"question":"标准问题","answer":"示例答案"}}。不要添加用户没提供的具体承诺。"""

DEDUP_SYSTEM = """判断新问题是否与待审列表中的某一条表达同一件事。只输出 JSON：
{{"matched_id":数字或null}}。只有含义相同才能合并，相近但答案不同返回 null。"""


def evidence_snapshot(evidence: list[dict]) -> list[dict]:
    """收敛检索结果为可审阅、可长期保存的轻量 JSON。"""
    return [
        {
            "chunk_id": item.get("chunk_id"),
            "question": str(item.get("question") or ""),
            "answer": str(item.get("answer") or "")[:200],
            "score": item.get("score"),
            "section_path": item.get("section_path"),
        }
        for item in evidence
    ]


def save_retrieval_snapshot(
    factory: sessionmaker[Session] | None,
    conversation_id: int | None,
    query: str,
    evidence: list[dict],
) -> list[dict]:
    snapshot = evidence_snapshot(evidence)
    if factory is None or not conversation_id:
        return snapshot
    with factory() as session:
        from app.models import Conversation

        if session.get(Conversation, conversation_id) is None:
            return snapshot
        session.add(RetrievalSnapshot(
            conversation_id=conversation_id,
            query=query,
            chunks=snapshot,
        ))
        session.commit()
    return snapshot


def latest_retrieval_snapshot(
    session: Session, conversation_id: int, before=None, after=None
) -> list[dict] | None:
    stmt = (
        select(RetrievalSnapshot)
        .where(RetrievalSnapshot.conversation_id == conversation_id)
        .order_by(desc(RetrievalSnapshot.created_at), desc(RetrievalSnapshot.id))
        .limit(1)
    )
    if before is not None:
        stmt = stmt.where(RetrievalSnapshot.created_at <= before)
    if after is not None:
        stmt = stmt.where(RetrievalSnapshot.created_at >= after)
    row = session.scalar(stmt)
    return row.chunks if row is not None else None


def record_low_confidence(
    factory: sessionmaker[Session],
    *,
    conversation_id: int | None,
    raw_question: str,
    source: str,
    reason: str,
    retrieval_snapshot: list[dict] | None = None,
) -> int:
    with factory() as session:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason[:500],
            retrieval_snapshot=retrieval_snapshot,
        )
        session.add(row)
        session.commit()
        return row.id


def schedule_low_confidence(row_id: int) -> None:
    asyncio.create_task(process_low_confidence(row_id))


async def process_low_confidence(row_id: int, llm=None) -> int | None:
    """标准化原话、与待审队列语义查重并回填归并落点。"""
    from app.agent.parsing import ai_text, extract_json_object
    from app.config import get_settings
    from app.db import get_session_factory
    from app.knowledge.models import ReviewQueue
    from app.llm import build_chat_model

    factory = get_session_factory()
    with factory() as session:
        source = session.get(LowConfidenceQuestion, row_id)
        if source is None or source.matched_review_id is not None:
            return source.matched_review_id if source is not None else None
        raw_question = source.raw_question

    llm = llm or build_chat_model(get_settings())
    normalized = extract_json_object(ai_text(await llm.ainvoke([
        SystemMessage(content=NORMALIZE_SYSTEM),
        HumanMessage(content=raw_question),
    ]))) or {}
    std_question = str(normalized.get("question") or raw_question).strip()
    sample_answer = str(normalized.get("answer") or "请审核后补充准确答案。").strip()

    async with _merge_lock:
        with factory() as session:
            source = session.get(LowConfidenceQuestion, row_id)
            if source is None or source.matched_review_id is not None:
                return source.matched_review_id if source is not None else None
            pending = session.scalars(
                select(ReviewQueue)
                .where(ReviewQueue.status == "待审")
                .order_by(ReviewQueue.id)
            ).all()
            exact = next((item for item in pending if item.std_question == std_question), None)
            candidates = [{"id": item.id, "question": item.std_question} for item in pending]

        matched_id = exact.id if exact is not None else None
        if matched_id is None and candidates:
            dedup = extract_json_object(ai_text(await llm.ainvoke([
                SystemMessage(content=DEDUP_SYSTEM),
                HumanMessage(content=(
                    f"新问题：{std_question}\n待审列表：{candidates}"
                )),
            ]))) or {}
            candidate_id = dedup.get("matched_id")
            if candidate_id in {item["id"] for item in candidates}:
                matched_id = int(candidate_id)

        with factory() as session:
            source = session.get(LowConfidenceQuestion, row_id)
            review = None
            if matched_id is not None:
                review = session.scalar(
                    select(ReviewQueue)
                    .where(
                        ReviewQueue.id == matched_id,
                        ReviewQueue.status == "待审",
                    )
                    .with_for_update()
                )
                if review is None:
                    matched_id = None
            if matched_id is None:
                review = ReviewQueue(
                    std_question=std_question,
                    sample_answer=sample_answer,
                )
                session.add(review)
                session.flush()
                matched_id = review.id
            else:
                review.occurrence_count += 1
            source.matched_review_id = matched_id
            session.commit()
    return matched_id


def publish_review_item(
    review_id: int,
    approved_answer: str,
    embedder=None,
    store=None,
) -> int:
    """把一条审核通过的答案发布到知识库，并只向量化这个新块。"""
    from app.db import get_session_factory
    from app.knowledge.embeddings import BgeM3Embedder
    from app.knowledge.models import ReviewQueue
    from app.knowledge.repository import insert_chunks, vectorize_pending
    from app.knowledge.splitter import SplitChunk
    from app.knowledge.vector_store import MilvusKnowledgeStore

    factory = get_session_factory()
    with factory() as session:
        review = session.scalar(
            select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update()
        )
        if review is None:
            raise LookupError(review_id)
        if review.status != "待审":
            raise ValueError(review.status)
        [chunk_id] = insert_chunks(session, [SplitChunk(
            category="人工审核",
            questions=review.std_question,
            answer=approved_answer,
            section_path=f"人工审核>{review.std_question[:40]}",
            content_type="faq",
            is_key_clause=0,
            source_order=0,
        )])
        review.status = "已通过"
        review.approved_answer = approved_answer
        session.commit()

    if store is None:
        store = MilvusKnowledgeStore()
        store.ensure_collection()
    vectorize_pending(
        factory,
        embedder or BgeM3Embedder.get(),
        store,
        row_ids=[chunk_id],
    )
    return chunk_id
