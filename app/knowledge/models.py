"""ch03 知识库 ORM：与 sql/schema_ch03.sql（用户 DDL 原文）逐列对应。

knowledge_chunks 是原文权威源 + 双写状态机载体；
qa_extraction_staging 是挖知识的离线中转暂存表。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.mysql import BIGINT, TINYINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class KnowledgeBase(DeclarativeBase):
    """独立 Base：知识库表与业务表分离，避免与 ch02 模型互相污染。"""


class KnowledgeChunk(KnowledgeBase):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255), index=True)
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(TINYINT(1), server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), server_default="pending", index=True
    )
    governance_status: Mapped[str] = mapped_column(
        Enum("active", "quarantined", "pending_review"),
        server_default="active",
        index=True,
    )
    governance_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    governed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(KnowledgeBase):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"), server_default="extracted", index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RetrievalSnapshot(KnowledgeBase):
    """每次知识检索的原始证据快照，供事后反馈回捞。"""

    __tablename__ = "retrieval_snapshots"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), index=True)
    query: Mapped[str] = mapped_column(Text)
    chunks: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class ReviewQueue(KnowledgeBase):
    """标准化、去重后的知识缺口待审队列。"""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    std_question: Mapped[str] = mapped_column(Text)
    sample_answer: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), server_default="数据飞轮", index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, server_default="1")
    status: Mapped[str] = mapped_column(
        Enum("待审", "已通过", "已驳回"), server_default="待审", index=True
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class LowConfidenceQuestion(KnowledgeBase):
    """低置信度问题池；ch09 增加证据快照与归并落点。"""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True, index=True)
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback"), index=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_snapshot: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class TopicClassification(KnowledgeBase):
    """ch10 低置信问题的离线多标签分类结果；question_id 唯一保证重跑幂等。"""

    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("low_confidence_questions.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    labels: Mapped[list[str]] = mapped_column(JSON)
    scores: Mapped[dict[str, float]] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(128), index=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class EvalRun(KnowledgeBase):
    """自动化评估每轮分数，供管理页按时间展示趋势。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    trigger: Mapped[str] = mapped_column(Enum("manual", "scheduled"), index=True)
    case_count: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class FaithCase(KnowledgeBase):
    """ch04 忠实度编造个案台账（DDL 原文零修改；uk_eval_id 一题一行跨轮幂等）。"""

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), server_default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决"), server_default="未解决", index=True
    )
    seen_count: Mapped[int] = mapped_column(BIGINT(unsigned=True), server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), index=True
    )
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
