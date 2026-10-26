"""HTTP 接缝：POST /api/chat 的 SSE 流式对话（ch02：工具轮 + DB 持久化）。

依赖真 MySQL 容器（落库断言），容器不在时自动跳过。
"""

import json

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import get_session_factory
from app.models import Conversation, Message
from langchain_core.messages import AIMessage
from tests.conftest import make_client
from tests.fakes import FakeChatLLM, make_stub_registry


def parse_sse(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def load_rows(conversation_id: int) -> list[Message]:
    with get_session_factory()() as s:
        return list(
            s.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            ).all()
        )


@pytest.fixture
def client(fake_llm, require_db):
    return make_client(fake_llm, registry=make_stub_registry())


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_tool_round_streams_meta_tool_frames_and_persists(client, fake_llm):
    fake_llm.first = AIMessage(
        content="",
        tool_calls=[
            {"name": "stub_lookup", "args": {"order_id": "1001"}, "id": "call_1", "type": "tool_call"}
        ],
    )
    with client.stream(
        "POST", "/api/chat", json={"user_id": "t-user", "message": "订单1001的物流到哪了"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        body = "".join(resp.iter_text())

    events = parse_sse(body)
    # 流首帧 meta：会话 id，供页面续聊
    assert events[0]["type"] == "meta"
    conv_id = events[0]["conversation_id"]
    assert isinstance(conv_id, int)
    # 工具状态帧 + 流式 deltas + done 收尾
    tool_events = [e for e in events if e["type"] == "tool"]
    assert [e["tool"] for e in tool_events] == ["stub_lookup"]
    deltas = "".join(e["content"] for e in events if e["type"] == "delta")
    assert deltas == fake_llm.stream_text
    assert events[-1]["type"] == "done"

    # 落库：user / assistant(tool_calls) / tool / assistant(最终)
    rows = load_rows(conv_id)
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[0].content == "订单1001的物流到哪了"
    assert rows[1].tool_calls[0]["name"] == "stub_lookup"
    assert rows[1].tool_calls[0]["id"] == "call_1"
    assert rows[2].tool_call_id == "call_1"
    assert "stub" in rows[2].content
    assert rows[3].content == fake_llm.stream_text


def test_no_tool_call_pseudo_streams_and_persists(client, fake_llm):
    fake_llm.first = AIMessage(content="直接回答不调工具")
    with client.stream(
        "POST", "/api/chat", json={"user_id": "t-user", "message": "你好"}
    ) as resp:
        body = "".join(resp.iter_text())

    events = parse_sse(body)
    assert events[0]["type"] == "meta"
    conv_id = events[0]["conversation_id"]
    assert [e for e in events if e["type"] == "tool"] == []
    deltas = "".join(e["content"] for e in events if e["type"] == "delta")
    assert deltas == "直接回答不调工具"

    rows = load_rows(conv_id)
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].tool_calls is None


def test_second_turn_carries_first_turn_context(client, fake_llm):
    fake_llm.first = AIMessage(content="你们支持7天无理由退货")
    with client.stream(
        "POST", "/api/chat", json={"user_id": "t-user", "message": "几天无理由退货"}
    ) as resp:
        body = "".join(resp.iter_text())
    conv_id = parse_sse(body)[0]["conversation_id"]

    # 第二轮：同一会话，模型第一调应看到第一轮的 user+assistant
    fake_llm.first = AIMessage(content="运费买家承担")
    fake_llm.seen.clear()
    with client.stream(
        "POST", "/api/chat",
        json={"user_id": "t-user", "message": "那运费谁出", "conversation_id": conv_id},
    ) as resp:
        resp.read()

    seen_texts = ["".join(str(m.content) for m in call) for call in fake_llm.seen]
    assert any("几天无理由退货" in t and "7天无理由退货" in t for t in seen_texts)


def test_unknown_conversation_id_creates_new(client, fake_llm):
    # 传不存在的 conversation_id：服务端新建会话（meta 帧给出新 id）
    with client.stream(
        "POST", "/api/chat",
        json={"user_id": "t-user", "message": "你好", "conversation_id": 999999},
    ) as resp:
        body = "".join(resp.iter_text())
    events = parse_sse(body)
    assert events[0]["type"] == "meta"
    new_id = events[0]["conversation_id"]
    assert new_id != 999999
    with get_session_factory()() as s:
        assert s.get(Conversation, new_id) is not None


def test_invalid_message_rejected(client):
    resp = client.post("/api/chat", json={"user_id": "t-user", "message": ""})
    assert resp.status_code == 422


def test_convergence_empty_stream_retries_and_closes_round(client, fake_llm):
    # 第二调吐空流（deepseek 不收敛行为）→ 非流式重试成功 → 回答落库、回合闭合
    fake_llm.first = AIMessage(
        content="",
        tool_calls=[
            {"name": "stub_lookup", "args": {"order_id": "1001"}, "id": "call_9", "type": "tool_call"}
        ],
    )
    fake_llm.stream_text = ""  # 收敛流式失败
    fake_llm.retry_results = [AIMessage(content="兜底重试的回答")]

    with client.stream(
        "POST", "/api/chat", json={"user_id": "t-user", "message": "邮费是多少"}
    ) as resp:
        body = "".join(resp.iter_text())
    events = parse_sse(body)
    conv_id = events[0]["conversation_id"]
    deltas = "".join(e["content"] for e in events if e["type"] == "delta")
    assert deltas == "兜底重试的回答"

    rows = load_rows(conv_id)
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[3].content == "兜底重试的回答"  # 回合闭合
    # 重试的 ainvoke 收到过收敛硬指令
    assert any("不要再调用工具" in str(m.content) for m in fake_llm.seen[-1])


def test_convergence_double_failure_persists_fallback(client, fake_llm):
    # 收敛流式空 + 重试也空 → 兜底文案吐给用户且落库（永不悬死、回合必闭合）
    fake_llm.first = AIMessage(
        content="",
        tool_calls=[
            {"name": "stub_lookup", "args": {"order_id": "1001"}, "id": "call_10", "type": "tool_call"}
        ],
    )
    fake_llm.stream_text = ""
    fake_llm.retry_results = [AIMessage(content="")]

    with client.stream(
        "POST", "/api/chat", json={"user_id": "t-user", "message": "邮费是多少"}
    ) as resp:
        body = "".join(resp.iter_text())
    events = parse_sse(body)
    conv_id = events[0]["conversation_id"]
    deltas = "".join(e["content"] for e in events if e["type"] == "delta")
    assert "抱歉" in deltas

    rows = load_rows(conv_id)
    assert rows[-1].role == "assistant"
    assert "抱歉" in rows[-1].content
