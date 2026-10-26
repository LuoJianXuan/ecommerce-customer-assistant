"""ch05 图全链路测试：fake deps 走真图（InMemorySaver），验证四出口分流。"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import (
    CHITCHAT_REPLY,
    GATE_FALLBACK_REPLY,
    AgentDeps,
    build_graph,
)


class ScriptLLM:
    """按 system prompt 标记分流应答：指代消解 prompt → 透传 JSON；分类 prompt → 意图 JSON。"""

    def __init__(self, intent="闲聊", final_text="好的，帮您查到了～"):
        self.intent = intent
        self.final_text = final_text
        self.invoke_calls = 0
        self.astream_calls = 0

    def _reply(self, messages):
        from langchain_core.messages import HumanMessage

        head = str(messages[0].content) if messages else ""
        if "指代消解" in head:
            last = next(
                (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
            )
            return f'{{"resolved_query": "{last}", "rewritten": false}}'
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
        self.astream_calls += 1
        for i in range(0, len(self.final_text), 3):
            yield AIMessageChunk(content=self.final_text[i : i + 3])


async def run_graph(deps, message):
    graph = build_graph(deps).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-1"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)], "conversation_id": 1, "user_id": "t"},
        config,
    )
    return result, graph.get_state(config).values


async def test_chitchat_path_fixed_reply_two_llm_calls():
    llm = ScriptLLM(intent="闲聊")
    deps = AgentDeps(llm=llm, retriever=lambda q: [], session_factory=None)
    result, _ = await run_graph(deps, "你好呀")
    assert result["final_reply"] == CHITCHAT_REPLY
    assert result["resolved_query"] == "你好呀"  # 首轮无历史：透传、零消解调用
    assert llm.invoke_calls == 1   # 只有 classify 一次（消解零调用）
    assert llm.astream_calls == 0  # Agent 没被走到


async def test_complaint_path_suggestions_no_backend_action():
    llm = ScriptLLM(intent="投诉")
    deps = AgentDeps(llm=llm, retriever=lambda q: [], session_factory=None)
    result, _ = await run_graph(deps, "太气人了我要投诉")
    assert [s["label"] for s in result["suggestions"]] == ["转人工", "建工单"]
    assert llm.astream_calls == 0  # 不进 Agent


async def test_knowledge_weak_evidence_blocked_by_gate():
    llm = ScriptLLM(intent="商品咨询", final_text="不该被生成")
    weak = [{"question": "q", "answer": "a", "score": 0.1}]
    deps = AgentDeps(llm=llm, retriever=lambda q: weak, session_factory=None, gate_threshold=0.35)
    result, _ = await run_graph(deps, "这个多少钱")
    assert result["final_reply"] == GATE_FALLBACK_REPLY
    assert llm.astream_calls == 0       # 证据弱不进 Agent
    assert "gate:block" in result["node_trace"]
    assert "retrieve" in result["node_trace"]  # 验收1：强制检索被走到


async def test_knowledge_strong_evidence_reaches_agent():
    llm = ScriptLLM(intent="商品咨询", final_text="支持 7 天无理由退货 [1]")
    strong = [{"question": "退货政策", "answer": "7天无理由", "score": 0.9, "chunk_id": 1}]
    deps = AgentDeps(llm=llm, retriever=lambda q: strong, session_factory=None)
    result, state = await run_graph(deps, "退货政策是什么")
    assert result["final_reply"].startswith("支持 7 天无理由退货")
    assert llm.astream_calls == 1  # Agent 走到且收敛
    assert state["retrieval_evidence"][0]["score"] == 0.9


async def test_agent_business_path_skips_retrieval():
    llm = ScriptLLM(intent="物流", final_text="您的包裹正在派送中～")
    calls = []
    def spy_retriever(q):
        calls.append(q)
        return []
    deps = AgentDeps(llm=llm, retriever=spy_retriever, session_factory=None)
    result, _ = await run_graph(deps, "订单1001到哪了")
    assert result["final_reply"] == "您的包裹正在派送中～"
    assert calls == []                     # 业务数据类不预检索
    assert "retrieve" not in result["node_trace"]
    assert any(t.startswith("agent:") for t in result["node_trace"])
