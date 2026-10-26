"""ch06 指代消解+改写接缝（TDD）：JSON 解析、空历史零调用透传、失败降级、图节点接线。

全部 fake LLM，不碰真模型。
"""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.coref import CorefResult, aresolve, parse_coref_json
from app.agent.graph import AgentDeps, resolve_refs_node


class FakeAsyncLLM:
    """按序回放 ainvoke 应答；耗尽后重复最后一条。记录每次看到的 messages。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list] = []

    async def ainvoke(self, messages, config=None):
        self.seen.append(list(messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=reply)


# ---------- parse_coref_json（纯函数） ----------

def test_parse_plain_json():
    got = parse_coref_json('{"resolved_query": "我买的电饭煲能退货吗", "rewritten": true}')
    assert got == CorefResult("我买的电饭煲能退货吗", True)


def test_parse_fenced_json():
    raw = "```json\n{\"resolved_query\": \"订单什么时候发货\", \"rewritten\": false}\n```"
    assert parse_coref_json(raw) == CorefResult("订单什么时候发货", False)


def test_parse_json_wrapped_in_prose():
    raw = "好的：{\"resolved_query\": \"耳机能退吗\", \"rewritten\": true} 请查收"
    assert parse_coref_json(raw) == CorefResult("耳机能退吗", True)


def test_parse_missing_rewritten_defaults_false():
    assert parse_coref_json('{"resolved_query": "原句"}') == CorefResult("原句", False)


def test_parse_string_false_rewritten_is_false():
    # 布尔陷阱：模型回字符串 "false"，bool("false") 是 True，必须按内容判
    got = parse_coref_json('{"resolved_query": "x", "rewritten": "false"}')
    assert got is not None and got.rewritten is False


def test_parse_empty_or_missing_query_returns_none():
    assert parse_coref_json('{"resolved_query": "", "rewritten": true}') is None
    assert parse_coref_json('{"rewritten": true}') is None


def test_parse_garbage_returns_none():
    assert parse_coref_json("我不会") is None
    assert parse_coref_json("") is None


# ---------- aresolve ----------

async def test_resolve_no_history_passthrough_zero_llm():
    llm = FakeAsyncLLM("不该被调用")
    r = await aresolve(llm, [], "会员有什么权益")
    assert r == CorefResult("会员有什么权益", False, degraded=False)
    assert llm.calls == 0  # 无历史无需消解，零模型调用


async def test_resolve_with_history_llm_called_and_prompt_carries_context():
    llm = FakeAsyncLLM('{"resolved_query": "我买的空气净化器能退货吗", "rewritten": true}')
    history = [HumanMessage("我上周买的空气净化器到哪了"), AIMessage("您的订单已发货")]
    r = await aresolve(llm, history, "这个能退吗")
    assert r.resolved_query == "我买的空气净化器能退货吗"
    assert llm.calls == 1
    joined = "".join(str(m.content) for m in llm.seen[0])
    assert "空气净化器" in joined and "这个能退吗" in joined  # 历史与最新句都进 prompt


async def test_resolve_parse_fail_falls_back_passthrough():
    llm = FakeAsyncLLM("胡言乱语")
    r = await aresolve(llm, [HumanMessage("x"), AIMessage("y")], "这个能退吗")
    assert r == CorefResult("这个能退吗", False, degraded=True)


async def test_resolve_history_drops_empty_tool_messages():
    llm = FakeAsyncLLM('{"resolved_query": "耳机电池能用多久", "rewritten": true}')
    from langchain_core.messages import ToolMessage

    history = [
        HumanMessage("这个耳机防水吗"),
        AIMessage(""),  # 工具步空内容
        ToolMessage(content='{"stock": 1}', tool_call_id="t1"),
        AIMessage("支持 IPX5 防水"),
    ]
    await aresolve(llm, history, "它电池能用多久")
    joined = "".join(str(m.content) for m in llm.seen[0])
    assert "stock" not in joined  # 工具消息与空内容不进对话转写


# ---------- 图节点 ----------

def make_deps(llm):
    return AgentDeps(llm=llm, retriever=lambda q: [], session_factory=None)


async def test_resolve_refs_node_writes_resolved_query():
    llm = FakeAsyncLLM('{"resolved_query": "我买的空气净化器能退货吗", "rewritten": true}')
    state = {
        "messages": [
            HumanMessage("我上周买的空气净化器到哪了"),
            AIMessage("已发货"),
            HumanMessage("这个能退吗"),
        ]
    }
    out = await resolve_refs_node(make_deps(llm), state)
    assert out["resolved_query"] == "我买的空气净化器能退货吗"
    assert out["node_trace"] == ["resolve_refs"]


async def test_resolve_refs_node_degrades_to_utterance():
    llm = FakeAsyncLLM("不是 JSON")
    state = {"messages": [HumanMessage("x"), AIMessage("y"), HumanMessage("它能退吗")]}
    out = await resolve_refs_node(make_deps(llm), state)
    assert out["resolved_query"] == "它能退吗"
