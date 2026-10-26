"""ch06 Query 扩写接缝（TDD）：JSON 解析、去重限量、失败降级单查询。

只在退款/售后子流程的政策检索前触发（高频低容错场景），简单 FAQ 不扩。
"""

from langchain_core.messages import AIMessage

from app.agent.query_expand import aexpand, parse_expand_json


class FakeAsyncLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list] = []

    async def ainvoke(self, messages, config=None):
        self.seen.append(list(messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=reply)


def test_parse_plain_list():
    got = parse_expand_json('{"queries": ["能退吗", "退货政策条件", "退款流程"]}')
    assert got == ["能退吗", "退货政策条件", "退款流程"]


def test_parse_fenced_json():
    raw = '```json\n{"queries": ["a", "b"]}\n```'
    assert parse_expand_json(raw) == ["a", "b"]


def test_parse_wrapped_in_prose():
    raw = '扩写如下：{"queries": ["退货条件", "退款时效"]} 请使用'
    assert parse_expand_json(raw) == ["退货条件", "退款时效"]


def test_parse_dedupes_preserving_order():
    got = parse_expand_json('{"queries": ["a", "b", "a", "c"]}')
    assert got == ["a", "b", "c"]


def test_parse_caps_at_max_queries():
    got = parse_expand_json('{"queries": ["1", "2", "3", "4", "5", "6"]}')
    assert got == ["1", "2", "3", "4"]  # MAX_QUERIES=4：原始+最多3个变体


def test_parse_rejects_bad_shapes():
    assert parse_expand_json('{"queries": "不是数组"}') is None
    assert parse_expand_json('{"queries": []}') is None
    assert parse_expand_json('{"other": 1}') is None
    assert parse_expand_json("不会") is None
    assert parse_expand_json("") is None


async def test_expand_happy():
    llm = FakeAsyncLLM('{"queries": ["耳机能退吗", "退货政策条件", "退款到账时效"]}')
    got = await aexpand(llm, "耳机能退吗")
    assert got == ["耳机能退吗", "退货政策条件", "退款到账时效"]
    assert llm.calls == 1


async def test_expand_parse_fail_falls_back_single_query():
    llm = FakeAsyncLLM("胡言乱语")
    got = await aexpand(llm, "耳机能退吗")
    assert got == ["耳机能退吗"]  # 检索永远有路可走


async def test_expand_empty_query_skips_llm():
    llm = FakeAsyncLLM("不该被调用")
    assert await aexpand(llm, "  ") == ["  "]
    assert llm.calls == 0
