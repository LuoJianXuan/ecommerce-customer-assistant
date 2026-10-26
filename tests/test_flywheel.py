"""ch09 数据飞轮公共接缝：用户反馈与审核队列 API。"""

import asyncio
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import delete, func, select

from app.db import get_session_factory
from app.knowledge.flywheel import latest_retrieval_snapshot, process_low_confidence, record_low_confidence
from app.knowledge.flywheel import publish_review_item
from app.knowledge.models import KnowledgeChunk, LowConfidenceQuestion, RetrievalSnapshot, ReviewQueue
from app.models import Conversation, Message
from tests.fakes import FakeChatLLM


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeStore:
    def __init__(self):
        self.rows = {}

    def upsert(self, vector_id, chunk_id, text, category, vector):
        self.rows[vector_id] = {"chunk_id": chunk_id, "text": text, "category": category}


class BrokenStore(FakeStore):
    def upsert(self, vector_id, chunk_id, text, category, vector):
        raise RuntimeError("milvus unavailable")


class ReviewProcessedDuringDedupLLM:
    def __init__(self, review_id: int):
        self.review_id = review_id
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content=(
                '{"question":"审核竞态标准问题？","answer":"请审核答案。"}'
            ))
        factory = get_session_factory()
        with factory() as session:
            review = session.get(ReviewQueue, self.review_id)
            review.status = "已通过"
            session.commit()
        return AIMessage(content=f'{{"matched_id":{self.review_id}}}')


def test_latest_snapshot_can_be_limited_to_current_turn(require_db):
    factory = get_session_factory()
    with factory() as session:
        conv = Conversation(user_id="ch09-snapshot-turn-test")
        session.add(conv)
        session.flush()
        old = RetrievalSnapshot(
            conversation_id=conv.id,
            query="上一轮",
            chunks=[{"chunk_id": 1}],
        )
        current = RetrievalSnapshot(
            conversation_id=conv.id,
            query="这一轮",
            chunks=[{"chunk_id": 2}],
        )
        session.add_all([old, current])
        session.flush()
        old.created_at = datetime.now() - timedelta(minutes=5)
        current.created_at = datetime.now()
        session.commit()
        conv_id = conv.id

    try:
        with factory() as session:
            assert latest_retrieval_snapshot(
                session,
                conv_id,
                after=datetime.now() - timedelta(minutes=1),
            ) == [{"chunk_id": 2}]
    finally:
        with factory() as session:
            session.execute(delete(RetrievalSnapshot).where(
                RetrievalSnapshot.conversation_id == conv_id
            ))
            session.execute(delete(Conversation).where(Conversation.id == conv_id))
            session.commit()

def test_negative_feedback_queues_the_question(client, require_db, monkeypatch):
    monkeypatch.setattr("app.agent.api.schedule_low_confidence", lambda _row_id: None)
    factory = get_session_factory()
    with factory() as session:
        conv = Conversation(user_id="ch09-feedback-test")
        session.add(conv)
        session.flush()
        user = Message(conversation_id=conv.id, role="user", content="这个滤芯能用三年吗")
        assistant = Message(conversation_id=conv.id, role="assistant", content="可以的")
        session.add_all([user, assistant])
        session.commit()
        conv_id, assistant_id = conv.id, assistant.id

    try:
        resp = client.post(
            "/api/agent/feedback",
            json={
                "conversation_id": conv_id,
                "message_id": assistant_id,
                "helpful": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["raw_question"] == "这个滤芯能用三年吗"
        assert isinstance(body["low_confidence_id"], int)
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == conv_id
            ))
            session.execute(delete(Message).where(Message.conversation_id == conv_id))
            session.execute(delete(Conversation).where(Conversation.id == conv_id))
            session.commit()


def test_negative_feedback_rejects_unknown_message_id(client, require_db, monkeypatch):
    monkeypatch.setattr("app.agent.api.schedule_low_confidence", lambda _row_id: None)
    factory = get_session_factory()
    with factory() as session:
        conv = Conversation(user_id="ch09-feedback-missing-message")
        session.add(conv)
        session.flush()
        session.add_all([
            Message(conversation_id=conv.id, role="user", content="已有问题"),
            Message(conversation_id=conv.id, role="assistant", content="已有回答"),
        ])
        session.commit()
        conv_id = conv.id

    try:
        resp = client.post(
            "/api/agent/feedback",
            json={
                "conversation_id": conv_id,
                "message_id": 999999999,
                "helpful": False,
            },
        )
        assert resp.status_code == 422
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == conv_id
            ))
            session.execute(delete(Message).where(Message.conversation_id == conv_id))
            session.execute(delete(Conversation).where(Conversation.id == conv_id))
            session.commit()


def test_negative_feedback_does_not_reuse_previous_turn_snapshot(
    client, require_db, monkeypatch
):
    monkeypatch.setattr("app.agent.api.schedule_low_confidence", lambda _row_id: None)
    factory = get_session_factory()
    base = datetime.now().replace(microsecond=0) - timedelta(minutes=10)
    with factory() as session:
        conv = Conversation(user_id="ch09-feedback-snapshot-boundary")
        session.add(conv)
        session.flush()
        session.add_all([
            Message(
                conversation_id=conv.id,
                role="user",
                content="上一轮问题",
                created_at=base,
            ),
            RetrievalSnapshot(
                conversation_id=conv.id,
                query="上一轮问题",
                chunks=[{"chunk_id": 1}],
                created_at=base + timedelta(seconds=1),
            ),
            Message(
                conversation_id=conv.id,
                role="assistant",
                content="上一轮回答",
                created_at=base + timedelta(seconds=2),
            ),
        ])
        session.flush()
        current_user = Message(
            conversation_id=conv.id,
            role="user",
            content="这一轮没有检索",
            created_at=base + timedelta(seconds=3),
        )
        current_assistant = Message(
            conversation_id=conv.id,
            role="assistant",
            content="直接回答",
            created_at=base + timedelta(seconds=4),
        )
        session.add_all([current_user, current_assistant])
        session.commit()
        conv_id, assistant_id = conv.id, current_assistant.id

    try:
        resp = client.post(
            "/api/agent/feedback",
            json={
                "conversation_id": conv_id,
                "message_id": assistant_id,
                "helpful": False,
            },
        )
        assert resp.status_code == 200
        with factory() as session:
            row = session.get(LowConfidenceQuestion, resp.json()["low_confidence_id"])
            assert not row.retrieval_snapshot
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.conversation_id == conv_id
            ))
            session.execute(delete(RetrievalSnapshot).where(
                RetrievalSnapshot.conversation_id == conv_id
            ))
            session.execute(delete(Message).where(Message.conversation_id == conv_id))
            session.execute(delete(Conversation).where(Conversation.id == conv_id))
            session.commit()


async def test_low_confidence_is_standardized_into_review_queue(client, require_db):
    factory = get_session_factory()
    row_id = record_low_confidence(
        factory,
        conversation_id=None,
        raw_question="滤芯是不是能撑三年啊",
        source="self_check",
        reason="证据不足",
    )
    llm = FakeChatLLM(first=AIMessage(content=(
        '{"question":"净水器滤芯的更换周期是多久？",'
        '"answer":"不同滤芯周期不同，请按型号说明更换。"}'
    )))
    review_id = await process_low_confidence(row_id, llm=llm)

    try:
        resp = client.get(f"/api/review/items/{review_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["std_question"] == "净水器滤芯的更换周期是多久？"
        assert body["occurrence_count"] == 1
        assert body["raw_questions"][0]["raw_question"] == "滤芯是不是能撑三年啊"
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.id == row_id))
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()

async def test_semantic_duplicate_increments_existing_review_item(require_db):
    factory = get_session_factory()
    with factory() as session:
        review = ReviewQueue(
            std_question="净水器滤芯多久更换？",
            sample_answer="按滤芯类型定期更换。",
        )
        session.add(review)
        session.commit()
        review_id = review.id
    row_id = record_low_confidence(
        factory,
        conversation_id=None,
        raw_question="滤芯三年换一次行吗",
        source="user_feedback",
        reason="用户反馈未解决",
    )
    llm = FakeChatLLM(first=AIMessage(content=(
        '{"question":"滤芯应该多长时间换一次？","answer":"请按说明定期更换。"}'
    )))
    llm.retry_results = [AIMessage(content=f'{{"matched_id":{review_id}}}')]

    try:
        assert await process_low_confidence(row_id, llm=llm) == review_id
        with factory() as session:
            merged = session.get(ReviewQueue, review_id)
            source = session.get(LowConfidenceQuestion, row_id)
            assert merged.occurrence_count == 2
            assert source.matched_review_id == review_id
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.id == row_id))
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()


async def test_concurrent_normalization_creates_one_review_item(require_db):
    factory = get_session_factory()
    row_ids = [
        record_low_confidence(
            factory,
            conversation_id=None,
            raw_question=question,
            source="user_feedback",
            reason="用户反馈未解决",
        )
        for question in ("并发测试问题甲", "并发测试问题乙")
    ]
    llms = [
        FakeChatLLM(first=AIMessage(content=(
            '{"question":"唯一并发标准问题？","answer":"请审核答案。"}'
        )))
        for _ in row_ids
    ]
    try:
        review_ids = await asyncio.gather(*(
            process_low_confidence(row_id, llm=llm)
            for row_id, llm in zip(row_ids, llms)
        ))
        assert review_ids[0] == review_ids[1]
        with factory() as session:
            review = session.get(ReviewQueue, review_ids[0])
            assert review.occurrence_count == 2
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(
                LowConfidenceQuestion.id.in_(row_ids)
            ))
            session.execute(delete(ReviewQueue).where(
                ReviewQueue.std_question == "唯一并发标准问题？"
            ))
            session.commit()


async def test_dedup_does_not_merge_into_item_processed_while_llm_runs(require_db):
    factory = get_session_factory()
    with factory() as session:
        old = ReviewQueue(std_question="旧的待审问题？", sample_answer="旧答案")
        session.add(old)
        session.commit()
        old_id = old.id
    row_id = record_low_confidence(
        factory,
        conversation_id=None,
        raw_question="审核竞态原话",
        source="user_feedback",
        reason="用户反馈未解决",
    )
    new_id = await process_low_confidence(
        row_id,
        llm=ReviewProcessedDuringDedupLLM(old_id),
    )
    try:
        assert new_id != old_id
        with factory() as session:
            assert session.get(ReviewQueue, old_id).status == "已通过"
            assert session.get(ReviewQueue, new_id).status == "待审"
            assert session.get(LowConfidenceQuestion, row_id).matched_review_id == new_id
    finally:
        with factory() as session:
            session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.id == row_id))
            session.execute(delete(ReviewQueue).where(ReviewQueue.id.in_([old_id, new_id])))
            session.commit()


def test_reject_keeps_review_item_for_audit(client, require_db):
    factory = get_session_factory()
    with factory() as session:
        row = ReviewQueue(std_question="无效问题？", sample_answer="无")
        session.add(row)
        session.commit()
        review_id = row.id
    try:
        resp = client.post(f"/api/review/items/{review_id}/reject")
        assert resp.status_code == 200
        assert resp.json()["status"] == "已驳回"
        assert client.get(f"/api/review/items/{review_id}").json()["status"] == "已驳回"
    finally:
        with factory() as session:
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()


def test_reject_cannot_reverse_an_approved_item(client, require_db):
    factory = get_session_factory()
    with factory() as session:
        row = ReviewQueue(std_question="已通过问题？", sample_answer="答案", status="已通过")
        session.add(row)
        session.commit()
        review_id = row.id
    try:
        resp = client.post(f"/api/review/items/{review_id}/reject")
        assert resp.status_code == 409
        with factory() as session:
            assert session.get(ReviewQueue, review_id).status == "已通过"
    finally:
        with factory() as session:
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()


def test_approve_publishes_one_reviewed_answer_to_knowledge_base(require_db):
    factory = get_session_factory()
    with factory() as session:
        row = ReviewQueue(
            std_question="净水器滤芯多久更换？",
            sample_answer="待审核",
        )
        session.add(row)
        session.commit()
        review_id = row.id
    store = FakeStore()
    chunk_id = publish_review_item(
        review_id,
        approved_answer="PP 棉滤芯建议每 6 个月更换一次。",
        embedder=FakeEmbedder(),
        store=store,
    )
    try:
        with factory() as session:
            review = session.get(ReviewQueue, review_id)
            chunk = session.get(KnowledgeChunk, chunk_id)
            assert review.status == "已通过"
            assert chunk.questions == "净水器滤芯多久更换？"
            assert chunk.answer == "PP 棉滤芯建议每 6 个月更换一次。"
            assert chunk.vectorize_status == "done"
            assert f"kb-{chunk_id}" in store.rows
    finally:
        with factory() as session:
            session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id))
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()


def test_vector_failure_does_not_duplicate_approved_chunk(require_db):
    factory = get_session_factory()
    with factory() as session:
        row = ReviewQueue(std_question="失败重试不能重复？", sample_answer="待审核")
        session.add(row)
        session.commit()
        review_id = row.id
    with pytest.raises(RuntimeError, match="milvus unavailable"):
        publish_review_item(
            review_id,
            approved_answer="唯一答案",
            embedder=FakeEmbedder(),
            store=BrokenStore(),
        )
    try:
        with factory() as session:
            review = session.get(ReviewQueue, review_id)
            assert review.status == "已通过"
            assert session.scalar(select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.questions == "失败重试不能重复？"
            )) == 1
        with pytest.raises(ValueError, match="已通过"):
            publish_review_item(
                review_id,
                approved_answer="唯一答案",
                embedder=FakeEmbedder(),
                store=FakeStore(),
            )
    finally:
        with factory() as session:
            session.execute(delete(KnowledgeChunk).where(
                KnowledgeChunk.questions == "失败重试不能重复？"
            ))
            session.execute(delete(ReviewQueue).where(ReviewQueue.id == review_id))
            session.commit()
