"""ch07 摘要域（TDD）：批次构造、用量口径、后台任务生命周期（成功/失败/跳过/并发守门）。

真 MySQL（require_db）；LLM 用 fake。
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.context_summary as cs
from app.context_budget import ContextBudget
from app.db import get_session_factory
from app.models import Conversation, ConversationSummary


@pytest.fixture(autouse=True)
def _db(require_db):
    pass


class FakeAsyncLLM:
    def __init__(self, reply="用户咨询耳机 E-3（订单 SO-AB12-02）退货，诉求未解决。"):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        from langchain_core.messages import AIMessage as AM

        return AM(content=self.reply)


class BrokenLLM:
    async def ainvoke(self, messages, config=None):
        raise RuntimeError("上游挂了")


def H(text, key=None):
    return HumanMessage(content=text, id=str(key) if key is not None else None)


def A(text, tool_calls=None):
    return AIMessage(content=text, tool_calls=tool_calls or [])


def T(content, name="query_order"):
    return ToolMessage(content=content, tool_call_id="c1", name=name)


def cnt(text: str) -> int:
    return len(str(text))


def _msgs():
    return [
        H("我买的耳机 E-3 订单 SO-AB12-02 能退吗", 101),
        A("", [{"name": "query_order", "args": {}, "id": "c1"}]),
        T('{"status":"已发货"}'),
        A("支持七天无理由退货"),
        H("运费谁出", 102),
        A("非质量问题自理"),
        H("那算了不退了", 103),
        A("好的"),
    ]


def _new_conv(session) -> int:
    conv = Conversation(user_id="u-test-" + uuid.uuid4().hex[:12])
    session.add(conv)
    session.commit()
    return conv.id


# ---------- 批次构造与用量口径 ----------

def test_build_batch_text_carries_facts():
    batch = cs.layer2_batch(_msgs(), summary_upto=None, layer1_from=103)
    text = cs.build_batch_text(batch)
    assert "SO-AB12-02" in text and "耳机 E-3" in text  # 关键事实必须在批次里
    assert "query_order" in text  # 工具调用留痕
    assert "用户：" in text and "客服：" in text


def test_should_trigger_by_layer2_usage():
    budget = ContextBudget(history=100, layer1=70, layer2=30, window_side=1, fixed_overhead=1, peak=1)
    msgs = [H("很长的用户话" * 5, 101), A("很长的客服答复" * 5), H("问2", 102), A("答2")]
    # 101/102 都在层2,用量超 30
    trig, usage = cs.should_trigger(msgs, summary_upto=None, layer1_from=103,
                                    budget=budget, count_tokens=cnt)
    assert trig is True and usage > 30
    # 全在层1 则不触发
    trig2, _ = cs.should_trigger(msgs, summary_upto=None, layer1_from=101,
                                 budget=budget, count_tokens=cnt)
    assert trig2 is False


# ---------- 后台任务生命周期 ----------

async def test_summary_task_success_appends_segment_and_advances_anchor(caplog):
    factory = get_session_factory()
    with factory() as s:
        conv_id = _new_conv(s)
    llm = FakeAsyncLLM()
    with caplog.at_level("INFO", logger="summary"):
        await cs.run_summary_task(
            factory, llm, conv_id,
            upto=None, layer1_from=102,   # 快照边界:轮 101 进梗概
            messages=_msgs(), count_tokens=cnt,
        )
    with factory() as s:
        rows = s.query(ConversationSummary).filter_by(conversation_id=conv_id).all()
        conv = s.get(Conversation, conv_id)
        assert len(rows) == 1 and rows[0].seq == 1
        assert "SO-AB12-02" in rows[0].content
        assert conv.summary_upto_msg_id == 102  # 锚点推进到触发时刻的层1 起点
    assert any("summary done 第1段" in r.message for r in caplog.records)


async def test_summary_task_failure_keeps_anchor(caplog):
    factory = get_session_factory()
    with factory() as s:
        conv_id = _new_conv(s)
        s.get(Conversation, conv_id).summary_upto_msg_id = None
        s.commit()
    with caplog.at_level("ERROR", logger="summary"):
        await cs.run_summary_task(
            factory, BrokenLLM(), conv_id,
            upto=None, layer1_from=102, messages=_msgs(), count_tokens=cnt,
        )
    with factory() as s:
        assert s.query(ConversationSummary).filter_by(conversation_id=conv_id).count() == 0
        assert s.get(Conversation, conv_id).summary_upto_msg_id is None  # 失败锚点不动
    assert any("summary failed" in r.message for r in caplog.records)


async def test_summary_task_skips_empty_batch(caplog):
    factory = get_session_factory()
    with factory() as s:
        conv_id = _new_conv(s)
    llm = FakeAsyncLLM()
    with caplog.at_level("INFO", logger="summary"):
        await cs.run_summary_task(
            factory, llm, conv_id,
            upto=102, layer1_from=102,  # 层2 空
            messages=_msgs(), count_tokens=cnt,
        )
    assert llm.calls == 0
    assert any("skipped" in r.message for r in caplog.records)


async def test_summary_task_single_flight_per_conversation(caplog):
    factory = get_session_factory()
    with factory() as s:
        conv_id = _new_conv(s)
    cs._running.add(conv_id)  # 模拟已在跑
    llm = FakeAsyncLLM()
    with caplog.at_level("INFO", logger="summary"):
        await cs.run_summary_task(
            factory, llm, conv_id,
            upto=None, layer1_from=102, messages=_msgs(), count_tokens=cnt,
        )
    assert llm.calls == 0  # 并发守门:同会话同时只跑一个
    cs._running.discard(conv_id)
