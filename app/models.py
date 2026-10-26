"""ORM 模型：与 sql/schema*.sql（MySQL，建表以它为准）逐列对应。"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束"), server_default="进行中"
    )
    # ch07 上下文锚点（messages 表 user 行 id；NULL=未启用分层）
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), index=True)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), index=True)
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询"))
    status: Mapped[str] = mapped_column(Enum("待处理", "已处理"), server_default="待处理")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Order(Base):
    """ch06 订单：订单选择器的数据源（懒播种演示数据，归属 user_id）。"""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(32), unique=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    product_name: Mapped[str] = mapped_column(String(128))
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    status: Mapped[str] = mapped_column(
        Enum("待付款", "已付款", "已发货", "已签收", "已取消")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class RefundRequest(Base):
    """ch06 退款单：退款表单提交落库，原因从固定类目里选。"""

    __tablename__ = "refund_requests"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    refund_no: Mapped[str] = mapped_column(String(32), unique=True)
    order_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str] = mapped_column(
        Enum("七天无理由", "质量问题", "与描述不符", "少件漏发", "不想要了", "其他")
    )
    status: Mapped[str] = mapped_column(
        Enum("待审核", "已通过", "已驳回"), server_default="待审核"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ConversationSummary(Base):
    """ch07 会话摘要段：层2 批次压出的梗概，一段一追加、压完不重写；老段留库不注入。"""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    upto_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
    )


class ToolAuditLog(Base):
    """ch08 工具调用审计：成功/失败/超时/校验拦下/权限拒绝 全落；不挂外键，写失败不拦执行。"""

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"))
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
