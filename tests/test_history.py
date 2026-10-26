"""历史裁剪接缝：token 预算内保留最近消息（注入假计数器，手算样例为独立真值）。

计数器用 len(text)，样例可手算。history 只含 human/ai 消息，system 由模板另行注入、永不进裁剪。
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.history import trim_history


def counter_len(text: str) -> int:
    return len(text)


def test_history_within_budget_kept_intact():
    history = [HumanMessage("aaaa"), AIMessage("bb"), HumanMessage("cc")]
    # 总开销 4+2+2=8 <= 10，全部保留，且保持时间顺序
    assert trim_history(history, budget=10, count_tokens=counter_len) == history


def test_over_budget_drops_oldest_first():
    history = [HumanMessage("a" * 10), AIMessage("b"), HumanMessage("c")]
    # 预算 5：从最新往回装 "c"(1)+"b"(1)=2，再装 "a"*10 会到 12 超预算 -> 最旧的被裁
    assert trim_history(history, budget=5, count_tokens=counter_len) == history[1:]


def test_newest_message_always_survives_even_if_alone_over_budget():
    history = [HumanMessage("a"), AIMessage("x" * 100)]
    # 预算 5，最新的 a*i 消息自身就 100 token，仍至少保留它，保证对话可继续
    assert trim_history(history, budget=5, count_tokens=counter_len) == history[1:]


def test_empty_history_returns_empty():
    assert trim_history([], budget=10, count_tokens=counter_len) == []


# ---------- ch02：DB 行 → LangChain 消息重建 ----------

from app.history import drop_orphan_tool_prefix, rows_to_messages  # noqa: E402
from app.models import Message  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402


def _row(role, content=None, tool_calls=None, tool_call_id=None):
    return Message(
        conversation_id=1, role=role, content=content,
        tool_calls=tool_calls, tool_call_id=tool_call_id,
    )


def test_rows_to_messages_full_tool_round():
    rows = [
        _row("user", "订单1001的物流到哪了"),
        _row("assistant", None, tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}]),
        _row("tool", '{"status": "运输中"}', tool_call_id="c1"),
        _row("assistant", "顺丰运输中"),
    ]
    msgs = rows_to_messages(rows)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["name"] == "query_logistics"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "顺丰运输中"


def test_rows_to_messages_plain_turns():
    msgs = rows_to_messages([_row("user", "你好"), _row("assistant", "你好呀")])
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]
    assert msgs[1].tool_calls == []


def test_drop_orphan_tool_prefix_removes_leading_tool_messages():
    msgs = [
        ToolMessage(content="孤儿结果", tool_call_id="x"),
        HumanMessage(content="下一轮问题"),
    ]
    cleaned = drop_orphan_tool_prefix(msgs)
    assert [m.content for m in cleaned] == ["下一轮问题"]


def test_drop_orphan_tool_prefix_keeps_normal_window():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert drop_orphan_tool_prefix(msgs) == msgs


# ---------- ch02 阶段8：未闭合工具回合修复 ----------

from app.history import repair_tool_rounds  # noqa: E402


def test_repair_keeps_closed_tool_round():
    msgs = rows_to_messages(
        [
            _row("user", "查物流"),
            _row("assistant", None, tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
            _row("tool", "结果", tool_call_id="c1"),
            _row("assistant", "答案"),
        ]
    )
    assert repair_tool_rounds(msgs) == msgs


def test_repair_strips_unclosed_tool_call_with_content():
    msgs = rows_to_messages(
        [
            _row("user", "查物流"),
            _row("assistant", "部分回答", tool_calls=[{"name": "t", "args": {}, "id": "cx", "type": "tool_call"}]),
            _row("user", "下一句"),
        ]
    )
    out = repair_tool_rounds(msgs)
    assert isinstance(out[1], AIMessage) and out[1].tool_calls == []
    assert out[1].content == "部分回答"


def test_repair_drops_unclosed_tool_call_without_content():
    msgs = rows_to_messages(
        [
            _row("user", "邮费"),
            _row("assistant", None, tool_calls=[{"name": "query_faq", "args": {}, "id": "cx", "type": "tool_call"}]),
            _row("user", "hello"),
        ]
    )
    out = repair_tool_rounds(msgs)
    assert [m.content for m in out] == ["邮费", "hello"]
