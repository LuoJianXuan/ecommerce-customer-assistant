"""ch07 会话侧栏只读 API（TDD）：会话列表（新在前/首问预览/已摘要标记）与历史回载。"""

import uuid

import pytest

from app.db import get_session_factory
from app.models import Conversation, ConversationSummary, Message


@pytest.fixture(autouse=True)
def _db(require_db):
    pass


def _seed(user_id: str) -> int:
    factory = get_session_factory()
    with factory() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        s.commit()
        s.add(Message(conversation_id=conv.id, role="user", content="第一问:耳机能退吗"))
        s.add(Message(conversation_id=conv.id, role="assistant", content="可以退"))
        s.add(ConversationSummary(conversation_id=conv.id, seq=1, upto_msg_id=1, content="梗概A"))
        s.commit()
        return conv.id


def test_conversations_list_new_first_with_preview_and_summary_flag(client):
    user = "u-test-" + uuid.uuid4().hex[:12]
    first_id = _seed(user)
    factory = get_session_factory()
    with factory() as s:  # 更新的会话(新在前)
        conv2 = Conversation(user_id=user)
        s.add(conv2)
        s.commit()
        s.add(Message(conversation_id=conv2.id, role="user", content="第二通会话的首问"))
        s.commit()

    resp = client.get("/api/conversations", params={"user_id": user})
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert items[0]["id"] == conv2.id and items[1]["id"] == first_id  # 新在前
    by_id = {it["id"]: it for it in items}
    assert by_id[first_id]["preview"].startswith("第一问")
    assert by_id[first_id]["has_summary"] is True
    assert by_id[conv2.id]["has_summary"] is False


def test_conversation_messages_returns_user_assistant_only(client):
    user = "u-test-" + uuid.uuid4().hex[:12]
    conv_id = _seed(user)
    resp = client.get(f"/api/conversations/{conv_id}/messages", params={"user_id": user})
    assert resp.status_code == 200
    msgs = resp.json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"].startswith("第一问")


def test_conversation_messages_rejects_other_user(client):
    user = "u-test-" + uuid.uuid4().hex[:12]
    conv_id = _seed(user)
    resp = client.get(
        f"/api/conversations/{conv_id}/messages",
        params={"user_id": "u-test-someone-else"},
    )
    assert resp.status_code == 404  # 不能读别人的会话


def test_conversation_messages_404_for_unknown(client):
    resp = client.get("/api/conversations/999999999/messages", params={"user_id": "u-test-x"})
    assert resp.status_code == 404
