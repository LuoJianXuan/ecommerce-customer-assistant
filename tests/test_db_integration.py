"""集成测试（真 MySQL 容器）：四表映射 CRUD 往返。容器不在时自动跳过。"""

import pytest
from sqlalchemy import select

from app.models import Conversation, Faq, Message, Ticket

pytestmark = pytest.mark.integration


@pytest.fixture
def db_session():
    from app.db import get_engine, get_session_factory

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(select(1))
    session = get_session_factory()()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


def test_conversation_message_roundtrip(db_session):
    conv = Conversation(user_id="it-user")
    db_session.add(conv)
    db_session.flush()

    db_session.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="订单1001的物流到哪了"),
            Message(
                conversation_id=conv.id,
                role="assistant",
                content=None,
                tool_calls=[
                    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_x1"}
                ],
            ),
            Message(
                conversation_id=conv.id,
                role="tool",
                content='{"carrier": "顺丰", "status": "运输中"}',
                tool_call_id="call_x1",
            ),
        ]
    )
    db_session.flush()

    rows = db_session.scalars(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)
    ).all()
    assert [r.role for r in rows] == ["user", "assistant", "tool"]
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[2].tool_call_id == "call_x1"
    assert rows[0].content == "订单1001的物流到哪了"


def test_faq_seeded_and_queryable(db_session):
    rows = db_session.scalars(select(Faq).where(Faq.question.like("%退货%"))).all()
    assert len(rows) >= 1
    assert "7天无理由" in rows[0].answer


def test_ticket_insert_with_conversation_fk(db_session):
    conv = Conversation(user_id="it-user")
    db_session.add(conv)
    db_session.flush()

    ticket = Ticket(
        ticket_no="T20260928001",
        conversation_id=conv.id,
        description="订单1001迟迟未发货",
        ticket_type="投诉",
    )
    db_session.add(ticket)
    db_session.flush()

    got = db_session.get(Ticket, "T20260928001")
    assert got.status == "待处理"
    assert got.conversation_id == conv.id


def test_conversation_default_status(db_session):
    conv = Conversation(user_id="it-user")
    db_session.add(conv)
    db_session.flush()
    assert conv.status == "进行中"
    assert conv.created_at is not None
