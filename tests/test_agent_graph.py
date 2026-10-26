"""ch05 图骨架接缝（TDD）：分流映射、置信度闸、投诉/闲聊短路、State 归并。

全部用 fake LLM/检索注入，不碰真模型与真 Milvus。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.graph import (
    GATE_FALLBACK_REPLY,
    AgentDeps,
    classify_intent,
    confidence_gate_node,
    complaint_node,
    INTENT_ROUTE,
    chitchat_node,
    evidence_confidence,
    retrieve_node,
    route_by_intent,
)


class FakeLLM:
    def __init__(self, text):
        self._text = text
        self.calls = 0
        self.seen: list[list] = []

    def invoke(self, messages, config=None):
        self.calls += 1
        self.seen.append(list(messages))
        return AIMessage(content=self._text)

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.seen.append(list(messages))
        return AIMessage(content=self._text)


def make_deps(llm=None, retriever=None, threshold=0.35):
    return AgentDeps(
        llm=llm or FakeLLM('{"intent": "闲聊", "confidence": 0.99}'),
        retriever=retriever or (lambda q: []),
        session_factory=None,
        gate_threshold=threshold,
        max_steps=5,
    )


# ---------- 分流映射（写死） ----------

def test_intent_route_mapping_eight_to_four():
    assert INTENT_ROUTE == {
        "商品咨询": "knowledge",
        "退款退货": "aftersale",  # ch06：确定性子流程
        "物流": "agent",
        "订单": "agent",
        "售后": "aftersale",     # ch06：与退款退货同一条子流程
        "投诉": "complaint",
        "闲聊": "chat",
        "其他": "agent",  # ch06：兜底类交主力 Agent
    }


def test_route_by_intent_default_falls_back_to_knowledge():
    assert route_by_intent({"intent": "物流"}) == "agent"
    assert route_by_intent({"intent": "未知意图"}) == "retrieve"  # 容错默认走知识类（检索节点）
    assert route_by_intent({"intent": "投诉"}) == "complaint"
    assert route_by_intent({}) == "retrieve"


# ---------- 意图识别 ----------

async def test_classify_intent_parses_json_with_confidence():
    deps = make_deps(llm=FakeLLM('{"intent": "退款退货", "confidence": 0.9}'))
    state = {"messages": [HumanMessage("我要退货")]}
    out = await classify_intent(deps, state)
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.9


async def test_classify_intent_prefers_resolved_query():
    llm = FakeLLM('{"intent": "退款退货", "confidence": 0.9}')
    deps = make_deps(llm=llm)
    state = {"messages": [HumanMessage("它能退吗")], "resolved_query": "订单 SO-1 的电饭煲能退货吗"}
    out = await classify_intent(deps, state)
    assert out["intent"] == "退款退货"
    # 分类器收到的是消解后的完整问句（判据单一，不喂全历史）
    assert llm.seen[-1][-1].content == "订单 SO-1 的电饭煲能退货吗"


async def test_classify_intent_invalid_json_falls_back_to_other():
    deps = make_deps(llm=FakeLLM("我不明白"))
    out = await classify_intent(deps, {"messages": [HumanMessage(" hi ")]})
    assert out["intent"] == "其他"  # 容错降级：拿不准归「其他」，不硬塞业务类
    assert out["intent_confidence"] == 0.0


# ---------- 置信度闸 ----------

def test_evidence_confidence_uses_top1_count_and_gap():
    evidence = [
        {"score": 0.72},
        {"score": 0.61},
        {"score": 0.44},
    ]
    result = evidence_confidence(evidence)
    assert result == {
        "confidence": 0.72,
        "top1": 0.72,
        "valid_count": 3,
        "gap": 0.11,
    }


def test_evidence_confidence_penalizes_sparse_and_close_evidence():
    result = evidence_confidence([{"score": 0.40}, {"score": 0.38}])
    assert result == {
        "confidence": 0.30,
        "top1": 0.40,
        "valid_count": 2,
        "gap": 0.02,
    }


def test_evidence_confidence_empty_is_zero():
    assert evidence_confidence([]) == {
        "confidence": 0.0,
        "top1": 0.0,
        "valid_count": 0,
        "gap": 0.0,
    }

def test_retrieve_node_prefers_resolved_query():
    seen = []

    def spy(q):
        seen.append(q)
        return []

    deps = make_deps(retriever=spy)
    state = {"messages": [HumanMessage("它能退吗")], "resolved_query": "我买的电饭煲能退货吗"}
    out = retrieve_node(deps, state)
    assert seen == ["我买的电饭煲能退货吗"]  # 检索用消解后的完整问句
    assert out["node_trace"] == ["retrieve"]


def test_gate_weak_evidence_blocks_agent_and_records():
    # Top-1 分数低于阈值 → 不进 Agent、给兜底话术
    weak = [{"question": "q", "answer": "a", "score": 0.10}]
    state = {"messages": [HumanMessage("x")], "retrieval_evidence": weak, "node_trace": []}
    deps = make_deps(threshold=0.35)
    out = confidence_gate_node(deps, state)
    assert out["gate_passed"] is False
    assert GATE_FALLBACK_REPLY[:6] in out["final_reply"]


def test_gate_strong_evidence_passes():
    strong = [{"question": "q", "answer": "a", "score": 0.9}]
    state = {"messages": [HumanMessage("x")], "retrieval_evidence": strong, "node_trace": []}
    deps = make_deps(threshold=0.35)
    out = confidence_gate_node(deps, state)
    assert out["gate_passed"] is True
    assert "final_reply" not in out  # 不写兜底


# ---------- 投诉出口（零模型） ----------

def test_complaint_node_fixed_reply_and_two_suggestions_zero_llm():
    llm = FakeLLM("不该被调用")
    deps = make_deps(llm=llm)
    state = {"messages": [HumanMessage("太气人了我要投诉")], "node_trace": []}
    out = complaint_node(deps, state)
    assert llm.calls == 0  # 零模型调用
    assert "安抚" in out["final_reply"] or "理解" in out["final_reply"]
    assert [s["label"] for s in out["suggestions"]] == ["转人工", "建工单"]
    ticket = next(s for s in out["suggestions"] if s["label"] == "建工单")
    assert "太气人了我要投诉" in ticket["payload"]["description"]  # 预生成工单描述


# ---------- 闲聊出口（零模型） ----------

def test_chitchat_node_fixed_reply_zero_llm():
    llm = FakeLLM("不该被调用")
    deps = make_deps(llm=llm)
    out = chitchat_node(deps, {"messages": [HumanMessage("你好呀")], "node_trace": []})
    assert llm.calls == 0
    assert out["final_reply"]  # 固定话术非空
    assert "suggestions" not in out
