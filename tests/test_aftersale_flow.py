"""ch06 退款/售后子流程（TDD）：interrupt 弹订单选择器 → resume 回填 →
Query 扩写+多路检索 → 主力 Agent 只做「这一单能不能退」终判。

真图 + InMemorySaver + ScriptLLM；订单走真 MySQL（懒播种），检索用 fake。
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import AgentDeps, build_graph
from app.db import get_session_factory


class ScriptLLM:
    """按 system prompt 标记分流：消解/扩写 → 透传 JSON；分类 → 意图 JSON；astream → 终答。"""

    def __init__(self, intent, final_text="这一单已发货且在七天窗口内，支持无理由退货 [1]"):
        self.intent = intent
        self.final_text = final_text
        self.invoke_calls = 0
        self.astream_seen: list[list] = []

    def _reply(self, messages):
        head = str(messages[0].content) if messages else ""
        last = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        if "指代消解" in head:
            return f'{{"resolved_query": "{last}", "rewritten": false}}'
        if "扩写器" in head:
            return f'{{"queries": ["{last}", "退货政策条件", "退款流程时效"]}}'
        return f'{{"intent": "{self.intent}", "confidence": 0.99}}'

    def invoke(self, messages, config=None):
        self.invoke_calls += 1
        return AIMessage(content=self._reply(messages))

    async def ainvoke(self, messages, config=None):
        self.invoke_calls += 1
        return AIMessage(content=self._reply(messages))

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages):
        self.astream_seen.append(list(messages))
        for i in range(0, len(self.final_text), 3):
            yield AIMessageChunk(content=self.final_text[i : i + 3])


@pytest.fixture(autouse=True)
def _db(require_db):
    pass


def make_deps(intent) -> tuple[AgentDeps, list[list[str]]]:
    llm = ScriptLLM(intent)
    multi_seen: list[list[str]] = []
    evidence = [{"chunk_id": 1, "question": "退货政策", "answer": "七天无理由", "score": 0.9}]
    deps = AgentDeps(
        llm=llm,
        retriever=lambda q: evidence,
        multi_retriever=lambda queries: (multi_seen.append(list(queries)), evidence)[1],
        session_factory=get_session_factory(),
    )
    return deps, multi_seen


def _new_user() -> str:
    return "u-test-" + uuid.uuid4().hex[:12]


async def test_mult_order_interrupt_then_resume_completes():
    deps, multi_seen = make_deps("退款退货")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-1"}}
    user = _new_user()

    result = await graph.ainvoke(
        {"messages": [HumanMessage("这个能退吗")], "conversation_id": 1, "user_id": user}, cfg
    )
    # 多单未选：子流程在订单选择器处暂停，本轮没有终答
    assert not result.get("final_reply")
    snap = graph.get_state(cfg)
    assert snap.next == ("aftersale_orders",)
    payload = snap.tasks[0].interrupts[0].value
    assert payload["type"] == "order_picker"
    orders = payload["orders"]
    assert len(orders) >= 3  # 懒播种给了演示订单

    # 点选第一单 → resume 接续：扩写+多路检索+agent 终判
    result2 = await graph.ainvoke(Command(resume={"order_id": orders[0]["id"]}), cfg)
    assert result2["final_reply"].startswith("这一单")
    assert result2["order_context"]["id"] == orders[0]["id"]
    assert result2["retrieval_evidence"][0]["chunk_id"] == 1
    assert result2["suggestions"][0]["action"] == "refund_form"
    assert result2["suggestions"][0]["payload"]["order_id"] == orders[0]["id"]
    assert "aftersale_orders" in result2["node_trace"]
    assert "aftersale_policy" in result2["node_trace"]
    # 多路检索收到的是扩写结果（透传原句 + 2 个变体）
    assert multi_seen[0] == ["这个能退吗", "退货政策条件", "退款流程时效"]
    # agent 收到订单上下文（终判指令 + 订单号）
    agent_prompt = "".join(str(m.content) for m in deps.llm.astream_seen[-1])
    assert "当前订单" in agent_prompt and orders[0]["order_no"] in agent_prompt


async def test_session_reuses_selected_order_no_second_interrupt():
    deps, _ = make_deps("退款退货")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-2"}}
    user = _new_user()

    first = await graph.ainvoke(
        {"messages": [HumanMessage("这个能退吗")], "conversation_id": 2, "user_id": user}, cfg
    )
    snap = graph.get_state(cfg)
    order_id = snap.tasks[0].interrupts[0].value["orders"][0]["id"]
    await graph.ainvoke(Command(resume={"order_id": order_id}), cfg)

    # 同会话再来一笔退款诉求：order_id 已绑定，不再弹选择器，直接走完
    second = await graph.ainvoke(
        {"messages": [HumanMessage("那运费谁出呢")], "user_id": user}, cfg
    )
    assert second["final_reply"].startswith("这一单")
    assert "aftersale_orders:invalid_pick" not in second["node_trace"]
    assert second["order_context"]["id"] == order_id


async def test_aftersale_intent_ends_with_ticket_suggestion():
    deps, _ = make_deps("售后")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-3"}}
    user = _new_user()

    result = await graph.ainvoke(
        {"messages": [HumanMessage("耳机坏了能换吗")], "conversation_id": 3, "user_id": user}, cfg
    )
    snap = graph.get_state(cfg)  # 多单：先弹选择器
    order_id = snap.tasks[0].interrupts[0].value["orders"][0]["id"]
    result = await graph.ainvoke(Command(resume={"order_id": order_id}), cfg)

    assert result["final_reply"].startswith("这一单")
    assert result["suggestions"][0]["action"] == "create_ticket"  # 售后收尾复用工单入口
    assert result["suggestions"][0]["payload"]["ticket_type"] == "售后"


async def test_no_orders_honest_reply(monkeypatch):
    import app.agent.orders as orders_mod

    monkeypatch.setattr(orders_mod, "get_or_seed_user_orders", lambda session, uid: [])
    deps, _ = make_deps("退款退货")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-4"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage("这个能退吗")], "conversation_id": 4, "user_id": _new_user()}, cfg
    )
    assert "查不到" in result["final_reply"]  # 无订单：如实说明，不瞎判
    assert "aftersale_orders:empty" in result["node_trace"]


async def test_named_order_ref_binds_directly_without_picker():
    """修复轮：点名订单号（消解问句里带 SO- 单号）→ 直接绑定,不弹选择器。"""
    from app.agent.orders import get_or_seed_user_orders

    user = _new_user()
    factory = get_session_factory()
    with factory() as s:
        orders = get_or_seed_user_orders(s, user)
    target = orders[1].order_no

    deps, multi_seen = make_deps("退款退货")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-5"}}
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(f"{target} 那单能退吗")],
            "conversation_id": 5,
            "user_id": user,
            "resolved_query": f"我的订单 {target} 能退货吗",
        },
        cfg,
    )
    assert result["final_reply"].startswith("这一单")  # 没有中断,一轮走完
    assert result["order_context"]["order_no"] == target
    assert result["suggestions"][0]["action"] == "refund_form"
    assert result["order_context"]["order_no"] == target


async def test_named_unknown_order_note_in_picker():
    """修复轮：点名了不存在的订单（如教程的 1001）→ 诚实提示进选择器 note,不是无声弹卡。"""
    deps, _ = make_deps("退款退货")
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "af-6"}}
    user = _new_user()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage("我订单1001到哪了"), HumanMessage("这个能退吗")],
            "conversation_id": 6,
            "user_id": user,
            "resolved_query": "我的订单1001能退货吗",
        },
        cfg,
    )
    assert not result.get("final_reply")
    snap = graph.get_state(cfg)
    payload = snap.tasks[0].interrupts[0].value
    assert payload["type"] == "order_picker"
    assert "1001" in payload["note"] and "没有找到" in payload["note"]
    assert len(payload["orders"]) >= 3
