"""ch07 会话侧栏只读 API：会话列表与历史回载（侧栏加载失败由前端静默降级）。

两接口都按 user_id 鉴属：会话详情校验归属，不能读别人的会话。
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import desc, select

from app.db import get_session_factory
from app.models import Conversation, ConversationSummary, Message

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

PREVIEW_LEN = 24


@router.get("")
def list_conversations(user_id: str) -> list[dict]:
    factory = get_session_factory()
    with factory() as session:
        convs = session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at), desc(Conversation.id))
            .limit(50)
        ).all()
        has_summary_ids = set(
            session.scalars(
                select(ConversationSummary.conversation_id).where(
                    ConversationSummary.conversation_id.in_([c.id for c in convs])
                )
            ).all()
        )
        previews: dict[int, str] = {}
        for c in convs:
            first = session.scalars(
                select(Message.content)
                .where(Message.conversation_id == c.id, Message.role == "user")
                .order_by(Message.id)
                .limit(1)
            ).first()
            previews[c.id] = (first or "")[:PREVIEW_LEN]
        return [
            {
                "id": c.id,
                "preview": previews.get(c.id, ""),
                "has_summary": c.id in has_summary_ids,
                "updated_at": c.updated_at.strftime("%m-%d %H:%M") if c.updated_at else "",
            }
            for c in convs
        ]


@router.get("/{conversation_id}/messages")
def conversation_messages(conversation_id: int, user_id: str) -> list[dict]:
    factory = get_session_factory()
    with factory() as session:
        conv = session.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user_id:
            raise HTTPException(404, f"会话不存在: {conversation_id}")
        rows = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        ).all()
        return [
            {"id": r.id, "role": r.role, "content": r.content or "", "created_at": r.created_at.strftime("%H:%M") if r.created_at else ""}
            for r in rows
            if r.role in ("user", "assistant")
        ]
