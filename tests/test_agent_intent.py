"""ch06 意图识别正式版接缝（TDD）：JSON 清洗校验、重试降级、两级升级链路。

全部用 fake LLM（异步 ainvoke 回放脚本应答），不碰真模型。
"""

from langchain_core.messages import AIMessage

from app.agent.intent import (
    FALLBACK_INTENT,
    INTENTS,
    IntentResult,
    aclassify,
    parse_intent_json,
)


class FakeAsyncLLM:
    """按序回放 ainvoke 应答；耗尽后重复最后一条（重试场景用）。"""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=reply)


# ---------- parse_intent_json（纯函数） ----------

def test_intents_contains_seven_business_plus_other():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
    assert FALLBACK_INTENT == "其他"


def test_parse_plain_json():
    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_fenced_json():
    raw = "```json\n{\"intent\": \"售后\", \"confidence\": 0.8}\n```"
    assert parse_intent_json(raw) == ("售后", 0.8)


def test_parse_json_wrapped_in_prose():
    raw = "好的，结果如下：{\"intent\": \"投诉\", \"confidence\": 0.75} 以上。"
    assert parse_intent_json(raw) == ("投诉", 0.75)


def test_parse_rejects_unknown_intent_name():
    assert parse_intent_json('{"intent": "退货", "confidence": 0.9}') is None


def test_parse_letter_intent_mapped():
    # 实测坑：模型把选择题字母当 intent 回（温度 0 确定性复读，重试救不回）
    assert parse_intent_json('{"intent": "H", "confidence": 0.97}') == ("其他", 0.97)
    assert parse_intent_json('{"intent": "a", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_letter_with_name_suffix():
    assert parse_intent_json('{"intent": "D. 退款退货", "confidence": 0.9}') == ("退款退货", 0.9)
    assert parse_intent_json('{"intent": "E、售后", "confidence": 0.9}') == ("售后", 0.9)


def test_parse_rejects_missing_confidence():
    assert parse_intent_json('{"intent": "物流"}') is None


def test_parse_rejects_garbage():
    assert parse_intent_json("我不明白") is None
    assert parse_intent_json("") is None
    assert parse_intent_json('["物流", 0.9]') is None  # 非对象


def test_parse_confidence_percent_coerced():
    assert parse_intent_json('{"intent": "闲聊", "confidence": 85}') == ("闲聊", 0.85)


def test_parse_confidence_out_of_range_clamped():
    got = parse_intent_json('{"intent": "闲聊", "confidence": 1.7}')
    assert got is not None and 0 < got[1] <= 1


def test_parse_confidence_string_number():
    assert parse_intent_json('{"intent": "订单", "confidence": "0.66"}') == ("订单", 0.66)


# ---------- aclassify（重试 / 兜底 / 两级升级） ----------

async def test_classify_happy_single_call():
    llm = FakeAsyncLLM('{"intent": "退款退货", "confidence": 0.93}')
    r = await aclassify(llm, "我要退货")
    assert r == IntentResult(intent="退款退货", confidence=0.93)
    assert llm.calls == 1


async def test_classify_retry_once_then_success():
    llm = FakeAsyncLLM("不是 JSON", '{"intent": "物流", "confidence": 0.9}')
    r = await aclassify(llm, "到哪了")
    assert r.intent == "物流"
    assert r.parse_failed is False
    assert llm.calls == 2


async def test_classify_both_fail_falls_back_to_other():
    llm = FakeAsyncLLM("胡言乱语")
    r = await aclassify(llm, "？？？")
    assert (r.intent, r.confidence, r.parse_failed) == ("其他", 0.0, True)
    assert llm.calls == 2  # 同参重试一次后仍失败才降级


async def test_escalation_low_confidence_invokes_big_model():
    small = FakeAsyncLLM('{"intent": "闲聊", "confidence": 0.4}')
    big = FakeAsyncLLM('{"intent": "售后", "confidence": 0.9}')
    r = await aclassify(big, "换货流程", small_llm=small, threshold=0.6)
    assert r.intent == "售后"  # 以大模型重判为准
    assert r.escalated is True
    assert small.calls == 1 and big.calls == 1


async def test_escalation_not_triggered_when_confident():
    small = FakeAsyncLLM('{"intent": "物流", "confidence": 0.95}')
    big = FakeAsyncLLM("大模型不该被调用")
    r = await aclassify(big, "到哪了", small_llm=small, threshold=0.6)
    assert r.intent == "物流"
    assert r.escalated is False
    assert big.calls == 0


async def test_escalation_triggered_by_parse_failure():
    small = FakeAsyncLLM("解析不了")
    big = FakeAsyncLLM('{"intent": "订单", "confidence": 0.9}')
    r = await aclassify(big, "订单还没发货", small_llm=small, threshold=0.6)
    assert r.intent == "订单"
    assert r.escalated is True


async def test_escalation_big_also_fails_falls_back_to_other():
    small = FakeAsyncLLM('{"intent": "闲聊", "confidence": 0.3}')
    big = FakeAsyncLLM("也解析不了")
    r = await aclassify(big, "？？", small_llm=small, threshold=0.6)
    assert (r.intent, r.parse_failed, r.escalated) == ("其他", True, True)
