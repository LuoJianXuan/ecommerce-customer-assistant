"""挖知识接缝（TDD）：分批 → LLM 抽 QA → 暂存 → 归一化去重 → 入库。

fake LLM / 真表（暂存与知识块落库走真 MySQL）。
"""

import pytest
from sqlalchemy import func, select

from app.knowledge.models import KnowledgeChunk, QaExtractionStaging, ReviewQueue
from app.knowledge.mine import (
    dedup_staging,
    extract_batch,
    is_rejection_answer,
    normalize_text,
)


class FakeMinerLLM:
    """按对话返回脚本化 QA 对的假抽取模型。"""

    def __init__(self, script: dict[str, list[dict]] | None = None):
        self.script = script or {}
        self.batches_seen: list[list[dict]] = []

    async def ainvoke(self, messages, config=None):
        # 最后一条消息是用户内容（含整批对话文本）；按脚本键匹配返回（透传全部键）
        self.batches_seen.append([{"role": m.type, "content": str(m.content)} for m in messages])
        user_text = str(messages[-1].content)
        for key, qas in self.script.items():
            if key in user_text:
                return [dict(q) for q in qas]
        return []


def conv_batch(conv_id: int, turns: list[tuple[str, str]]) -> dict:
    return {
        "conversation_id": conv_id,
        "messages": [{"role": r, "content": c} for r, c in turns],
    }


@pytest.fixture
def db(require_db):
    from app.db import get_session_factory

    factory = get_session_factory()
    with factory() as s:
        yield factory, s
        s.query(QaExtractionStaging).filter(QaExtractionStaging.batch_no.like("test-%")).delete()
        s.query(ReviewQueue).where(ReviewQueue.source == "对话挖掘-test").delete()
        s.query(KnowledgeChunk).where(KnowledgeChunk.category == "对话挖掘-test").delete()
        s.commit()


def test_normalize_text_fullwidth_and_whitespace():
    assert normalize_text("退货 怎么办？") == "退货怎么办"      # 去空白+剥句尾问号
    assert normalize_text("Ａｐｐ发票") == "App发票"            # 全角转半角
    assert normalize_text("  七天 无理由  ") == "七天无理由"


def test_rejection_answers_are_filtered():
    assert is_rejection_answer("抱歉，知识库暂未收录该邮费标准。")
    assert is_rejection_answer("平台暂未收录这一型号的参数，请稍后再试。")
    assert is_rejection_answer("当前信息不足，建议人工处理。")
    assert is_rejection_answer("这个问题我无法回答，建议联系人工客服。")
    assert is_rejection_answer("可以选择转人工，或建工单让专员处理。")
    assert is_rejection_answer(
        "目前系统没有足够的商品参数可以核实这项承诺。为了避免给出错误信息，"
        "这里先不做猜测；如果您仍需确认，建议联系人工客服，由专员进一步处理。"
    )
    assert not is_rejection_answer("偏远地区邮费为 12 元，满 99 元包邮。")
    assert not is_rejection_answer(
        "签收后48小时内拍照留证并联系客服，再从订单页申请售后；"
        "质量问题退货运费由平台承担，仓库验收后1-3个工作日退款。"
    )


def test_rejection_filter_keeps_informative_answers_with_handoff():
    assert not is_rejection_answer(
        "平台支持微信支付、支付宝、云闪付以及各主流银行卡付款，"
        "部分商品支持白条分期；花呗分期请在结算页查看或联系人工客服确认。"
    )
    assert not is_rejection_answer(
        "建工单前需提供具体问题，以及涉及的订单号或商品；"
        "提交后由人工客服跟进。"
    )


@pytest.mark.asyncio
async def test_extract_batch_stages_qa(db):
    factory, s = db
    llm = FakeMinerLLM(script={"退货": [{"conversation_id": 1, "question": "退货怎么操作", "answer": "订单页申请售后"}]})
    batch = [conv_batch(1, [("user", "我要退货"), ("assistant", "好的，请提供订单号")])]

    await extract_batch(s, llm, batch, batch_no="test-b1")

    rows = s.scalars(
        select(QaExtractionStaging).where(QaExtractionStaging.batch_no == "test-b1")
    ).all()
    assert len(rows) == 1
    assert rows[0].question == "退货怎么操作"
    assert rows[0].status == "extracted"
    assert rows[0].source_ref == "conv:1"


@pytest.mark.asyncio
async def test_extract_batch_drops_rejection_answer(db):
    _, s = db
    llm = FakeMinerLLM(script={"邮费": [
        {"conversation_id": 1, "question": "新疆邮费多少", "answer": "知识库暂未收录，建议咨询人工客服。"},
    ]})
    batch = [conv_batch(1, [("user", "新疆邮费多少"), ("assistant", "暂时查不到")])]

    assert await extract_batch(s, llm, batch, batch_no="test-reject") == 0
    assert s.scalar(select(func.count()).select_from(QaExtractionStaging).where(
        QaExtractionStaging.batch_no == "test-reject"
    )) == 0


def test_dedup_marks_kept_and_discarded(db):
    """注意：判重范围是全库 questions（真实行为），测试用独特文案避免与真实挖矿数据撞车。"""
    factory, s = db
    s.add_all(
        [
            QaExtractionStaging(batch_no="test-b1", source_ref="conv:1", question="测试问题甲怎么处理", answer="订单页申请售后"),
            QaExtractionStaging(batch_no="test-b2", source_ref="conv:2", question="测试问题甲 怎么处理？", answer="订单页申请售后"),  # 归一化后重复
            QaExtractionStaging(batch_no="test-b2", source_ref="conv:3", question="测试问题乙如何办理XYZ", answer="订单页申请开票"),
        ]
    )
    s.commit()

    kept = dedup_staging(s, category="对话挖掘-test")

    rows = {r.question: r.status for r in s.scalars(select(QaExtractionStaging).where(QaExtractionStaging.batch_no.like("test-%"))).all()}
    assert rows["测试问题甲怎么处理"] == "kept"
    assert rows["测试问题甲 怎么处理？"] == "discarded"  # 全半角/空白归一化后判重
    assert rows["测试问题乙如何办理XYZ"] == "kept"

    # kept 改道待审队列，不再把未审核答案直接写入知识库
    review = s.scalars(select(ReviewQueue).where(ReviewQueue.source == "对话挖掘-test")).all()
    assert {item.std_question for item in review} == {"测试问题甲怎么处理", "测试问题乙如何办理XYZ"}
    assert all(item.status == "待审" for item in review)
    assert s.scalar(select(func.count()).select_from(KnowledgeChunk).where(
        KnowledgeChunk.category == "对话挖掘-test"
    )) == 0
    assert len(kept) == 2
