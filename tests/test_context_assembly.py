"""ch07 三层装配引擎（TDD）：轮次切分、锚点分层、层2 折叠、层1 降级+trim 兜底、拼装顺序。

纯函数测试，不碰 DB/LLM；token 计数用「1 非 ASCII 字符=1 token」的确定口径。
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.context_assembly import (
    AssembledContext,
    assemble_context,
    build_background_message,
    classify_rounds,
    split_rounds,
)
from app.context_budget import ContextBudget

BUDGET = ContextBudget(history=1000, layer1=700, layer2=300, window_side=5000, fixed_overhead=1, peak=1)


def cnt(text: str) -> int:
    return len(str(text))  # 测试口径：1 字符 = 1 token


def H(text, key=None):
    return HumanMessage(content=text, id=str(key) if key is not None else None)


def A(text, tool_calls=None):
    return AIMessage(content=text, tool_calls=tool_calls or [])


def T(content, call_id="c1", name="query_order"):
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def TC(name="query_order", call_id="c1"):
    return {"name": name, "args": {}, "id": call_id}


# ---------- 轮次切分 ----------

def test_split_rounds_groups_by_human_boundary():
    msgs = [H("问1", 101), A("答1"), H("问2", 102), A("", [TC()]), T('{"ok":1}'), A("答2")]
    rounds = split_rounds(msgs)
    assert [r.key for r in rounds] == [101, 102]
    assert len(rounds[1].msgs) == 4  # 工具回合整轮归属


def test_split_rounds_marker_without_key_joins_flow():
    msgs = [H("问1", 101), A("答1"), H("（已选择订单：SO-1）"), A("答2")]
    rounds = split_rounds(msgs)
    assert [r.key for r in rounds] == [101, None]  # 无 id 的标记自成一轮,key=None


# ---------- 锚点分层 ----------

def test_classify_by_anchors():
    msgs = [H("问1", 101), A("答1"), H("问2", 102), A("答2"), H("问3", 103), A("答3")]
    rounds = split_rounds(msgs)
    l3, l2, l1 = classify_rounds(rounds, summary_upto=101, layer1_from=103)
    assert [r.key for r in l3] == [101]   # 已被摘要覆盖
    assert [r.key for r in l2] == [102]   # 截短层
    assert [r.key for r in l1] == [103]   # 原文层


def test_classify_no_anchors_all_layer1():
    msgs = [H("问1", 101), A("答1"), H("问2", 102), A("答2")]
    rounds = split_rounds(msgs)
    l3, l2, l1 = classify_rounds(rounds, summary_upto=None, layer1_from=None)
    assert l3 == [] and l2 == []
    assert [r.key for r in l1] == [101, 102]


# ---------- 拼装：顺序与背景条 ----------

def test_assemble_order_and_background_single_human_after_user():
    msgs = [
        H("问1", 101), A("很长的答复" * 50),
        H("问2", 102), A("答2"),
        H("这个能退吗", 103),
    ]
    out = assemble_context(
        system_prompt="人设", messages=msgs,
        summary_upto=None, layer1_from=102,  # 101=层2(截短),102起=层1原文
        summaries=["用户咨询耳机 SO-1 退货。"],
        evidence_text="[1] 退货政策:七天无理由",
        resolved_note="我的订单 SO-1 能退货吗",
        budget=BUDGET, count_tokens=cnt, assistant_head_chars=80,
    )
    kinds = [type(m) for m in out.messages]
    # 固定顺序:System → 层2(截短) → 层1(原文) → 背景条(单条 Human)
    assert kinds[0] is SystemMessage
    assert kinds[-1] is HumanMessage
    assert sum(isinstance(m, ToolMessage) for m in out.messages) == 0  # 无悬空工具消息
    bg = out.messages[-1]
    assert "SO-1" in bg.content and "七天无理由" in bg.content and "指代消解" in bg.content
    # 用户当前句在背景条之前、且是层1尾部
    assert out.messages[-2].content == "这个能退吗"
    # 层2 的用户原话不动、答复截短
    l2_human = next(m for m in out.messages if getattr(m, "content", "") == "问1")
    assert isinstance(l2_human, HumanMessage)


def test_assemble_layer2_folds_tool_round_without_protocol_break():
    msgs = [
        H("查下物流", 101), A("", [TC("query_logistics")]), T('{"轨迹":"很长" * 100}', name="query_logistics"), A("已发货"),
        H("谢谢", 102), A("不客气"),
    ]
    out = assemble_context(
        system_prompt="s", messages=msgs,
        summary_upto=None, layer1_from=102,  # 101 在层2
        summaries=[], evidence_text="", resolved_note=None,
        budget=BUDGET, count_tokens=cnt, assistant_head_chars=80,
    )
    # 层2 输出里:Human 原话 + 单条 AIMessage(截短+工具省略行),无 tool_calls/ToolMessage
    tool_ais = [m for m in out.messages
                if isinstance(m, AIMessage) and ("query_logistics" in str(m.content) or "省略" in str(m.content))]
    assert tool_ais, "工具回合应折叠成带省略标注的 AIMessage"
    assert all(not m.tool_calls for m in out.messages if isinstance(m, AIMessage))
    assert not any(isinstance(m, ToolMessage) for m in out.messages)


def test_assemble_no_background_when_all_empty():
    msgs = [H("你好", 101), A("你好呀")]
    out = assemble_context(
        system_prompt="s", messages=msgs, summary_upto=None, layer1_from=None,
        summaries=[], evidence_text="", resolved_note=None,
        budget=BUDGET, count_tokens=cnt, assistant_head_chars=80,
    )
    assert type(out.messages[-1]) is AIMessage  # 没有背景条
    assert out.new_layer1_from is None and out.degraded is False


# ---------- 层1 降级:只挪锚点 ----------

def test_layer1_degradation_moves_anchor_not_data():
    # 每轮 H(2字)+A(2字)=4 token;层1 预算 9 → 三轮 12>9,降一轮后 8≤9
    tiny = ContextBudget(history=12, layer1=9, layer2=3, window_side=1, fixed_overhead=1, peak=1)
    msgs = [H("问1", 101), A("答1"), H("问2", 102), A("答2"), H("问3", 103), A("答3")]
    out = assemble_context(
        system_prompt="s", messages=msgs, summary_upto=None, layer1_from=None,
        summaries=[], evidence_text="", resolved_note=None,
        budget=tiny, count_tokens=cnt, assistant_head_chars=80,
    )
    assert out.degraded is True
    assert out.new_layer1_from == 102  # 锚点前移到轮 102,轮 101 挪入层2(数据不动)
    assert out.round_count == 2
    # 挪入层2 的轮 101 仍在上下文(用户原话可见)
    assert any(getattr(m, "content", "") == "问1" for m in out.messages)
    assert out.layer1_tokens <= tiny.layer1


def test_layer1_keeps_at_least_one_round():
    tiny = ContextBudget(history=10, layer1=7, layer2=3, window_side=1, fixed_overhead=1, peak=1)
    msgs = [H("很长的问题" * 20, 101), A("很长的回答" * 20)]
    out = assemble_context(
        system_prompt="s", messages=msgs, summary_upto=None, layer1_from=None,
        summaries=[], evidence_text="", resolved_note=None,
        budget=tiny, count_tokens=cnt, assistant_head_chars=80,
    )
    assert out.round_count == 1  # 只有一轮时不降级,交给 trim_messages 兜底
    assert out.messages[-1] is not None


def test_trim_messages_fallback_for_giant_single_round():
    tiny = ContextBudget(history=10, layer1=7, layer2=3, window_side=1, fixed_overhead=1, peak=1)
    msgs = [H("很长的当前问题" * 30, 101)]  # 单轮且超预算
    out = assemble_context(
        system_prompt="s", messages=msgs, summary_upto=None, layer1_from=None,
        summaries=[], evidence_text="", resolved_note=None,
        budget=tiny, count_tokens=cnt, assistant_head_chars=80,
    )
    humans = [m for m in out.messages if isinstance(m, HumanMessage)]
    assert humans, "兜底后窗口仍应保留尾部(以 human 开头)"


# ---------- 背景条构造 ----------

def test_build_background_message_combines_and_handles_empty():
    bg = build_background_message(["段1", "段2"], "[1] 证据", "消解句")
    assert isinstance(bg, HumanMessage)
    assert "段1" in bg.content and "证据" in bg.content and "消解句" in bg.content
    assert build_background_message([], "", None) is None
    assert build_background_message(["段1"], "", None) is not None
