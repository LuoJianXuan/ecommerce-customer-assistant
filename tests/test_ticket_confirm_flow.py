"""ch08 建工单确认流（TDD,方案B）：挂起→确认落库 / 挂起→取消=权限拒绝 / 缺描述=校验拦下追问。

真图 + InMemorySaver + ScriptLLM（astream 脚本回放）;订单/工单走真 MySQL。
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import text

from app.agent.graph import AgentDeps, build_graph
from app.db import get_session_factory
from app.models import Conversation, Ticket


@pytest.fixture(autouse=True)
def _db(require_db):
    pass


class ScriptReActLLM:
    """astream 按脚本回放：先吐 tool_calls 的 AIMessage,再吐终答文本。"""

    def __init__(self, tool_call: dict | None, final_text: str = "已为您建好工单～"):
        self.tool_call = tool_call
        self.final_text = final_text
        self.astream_calls = 0

    def _reply(self):
        self.astream_calls += 1
        if self.astream_calls == 1 and self.tool_call is not None:
            return None  # 走 astream 的 chunk 流
        return AIMessage(content=self.final_text)

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        return self._reply()

    async def astream(self, messages):
        if self.tool_call is not None and getattr(self, "_first_done", False) is False:
            self._first_done = True
            base = {"id": self.tool_call.get("id", "call_t1"), "args": self.tool_call.get("args", {}),
                    "name": self.tool_call.get("name", "create_ticket")}
            yield AIMessageChunk(content="", tool_calls=[base])
            return
        for i in range(0, len(self.final_text), 3):
            yield AIMessageChunk(content=self.final_text[i : i + 3])


def _new_conv(session) -> tuple[int, str]:
    user = "u-test-" + uuid.uuid4().hex[:12]
    conv = Conversation(user_id=user)
    session.add(conv)
    session.commit()
    return conv.id, user


def make_deps(llm) -> AgentDeps:
    return AgentDeps(llm=llm, retriever=lambda q: [], session_factory=get_session_factory())


def ticket_count(factory, conv_id) -> int:
    with factory() as s:
        return s.execute(
            text("SELECT COUNT(*) FROM tickets WHERE conversation_id=:c"), {"c": conv_id}
        ).scalar()


def last_audit(factory, conv_id, tool="create_ticket"):
    with factory() as s:
        return s.execute(
            text("SELECT status, retry_count FROM tool_audit_logs WHERE conversation_id=:c "
                 "AND tool_name=:t ORDER BY id DESC LIMIT 1"),
            {"c": conv_id, "t": tool},
        ).first()


def _ticket_call(description="耳机坏了要返修", ticket_type="售后"):
    return {"name": "create_ticket", "args": {"description": description, "ticket_type": ticket_type},
            "id": "call_t1"}


async def test_confirm_path_creates_ticket_and_audits_success():
    factory = get_session_factory()
    with factory() as s:
        conv_id, user = _new_conv(s)
    llm = ScriptReActLLM(_ticket_call(), final_text="好的,工单已建:马上安排专员跟进。")
    graph = build_graph(make_deps(llm)).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tc-1"}}

    r1 = await graph.ainvoke(
        {"messages": [HumanMessage("帮我建个工单,耳机坏了要返修")], "conversation_id": conv_id, "user_id": user}, cfg
    )
    assert "agent:write_pending" in r1["node_trace"]
    assert not r1.get("final_reply")  # 挂起轮无终答
    snap = graph.get_state(cfg)
    preview = snap.tasks[0].interrupts[0].value
    assert preview["type"] == "ticket_confirm" and "耳机坏了" in preview["description"]

    r2 = await graph.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert "ticket_confirm:confirmed" in r2["node_trace"]
    assert r2["final_reply"]  # agent 重入收尾
    assert ticket_count(factory, conv_id) == 1
    assert last_audit(factory, conv_id) == ("成功", 0)


async def test_cancel_path_no_ticket_audits_permission_denied():
    factory = get_session_factory()
    with factory() as s:
        conv_id, user = _new_conv(s)
    llm = ScriptReActLLM(_ticket_call(), final_text="好的,已按您的要求取消。")
    graph = build_graph(make_deps(llm)).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tc-2"}}

    await graph.ainvoke(
        {"messages": [HumanMessage("帮我建个工单,耳机坏了")], "conversation_id": conv_id, "user_id": user}, cfg
    )
    r2 = await graph.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert "ticket_confirm:cancelled" in r2["node_trace"]
    assert ticket_count(factory, conv_id) == 0                      # 工单没建(验收⑤)
    assert last_audit(factory, conv_id) == ("权限拒绝", 0)           # 审计=权限拒绝(验收⑤)


async def test_missing_description_blocked_and_model_asks():
    """描述为空 → 引擎校验拦下回灌 → 模型(脚本)转向追问。"""
    factory = get_session_factory()
    with factory() as s:
        conv_id, user = _new_conv(s)
    llm = ScriptReActLLM(_ticket_call(description="  "), final_text="请问您遇到了什么问题?我需要具体描述才能建单。")
    graph = build_graph(make_deps(llm)).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tc-3"}}

    r1 = await graph.ainvoke(
        {"messages": [HumanMessage("帮我建个工单")], "conversation_id": conv_id, "user_id": user}, cfg
    )
    assert r1["final_reply"] and "什么问题" in r1["final_reply"]  # 拦下后模型追问,无挂起
    assert "agent:write_pending" not in r1["node_trace"]
    assert ticket_count(factory, conv_id) == 0
    assert last_audit(factory, conv_id) is not None and last_audit(factory, conv_id)[0] == "校验拦下"
