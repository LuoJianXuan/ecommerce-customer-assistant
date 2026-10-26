"""对话挖知识：历史会话分批 → LLM 抽 QA → 拒答过滤 → 去重 → 待审队列。

去重实现决策（计划 v4）：归一化精确去重（全半角统一+去空白），
比对范围=暂存表全量已抽 + 库内 questions；不做 LLM 语义判重（留给后续）。
"""

import json
import logging
import re
import unicodedata
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.models import KnowledgeChunk, QaExtractionStaging, ReviewQueue
from app.models import Conversation, Message

MINER_SYSTEM = """你是客服知识抽取器。从给定的客服对话中抽取有复用价值的问答对：
- 只抽"用户真实问过、客服真实答过"且答案明确有信息量的问题；寒暄、闲聊、未解决的问题不抽。
- 答案含拒答、无法回答、未收录、查不到、建议转人工等表述时一律不抽。
- question 用用户的原始问法（可轻度顺句），answer 用客服答案的精炼版（保留关键事实：时限/金额/步骤）。
- conversation_id 填该问答对所在的会话编号。
- 每段对话最多抽 3 对。没有可抽的就返回空数组。

以 JSON 数组返回：[{{"conversation_id": 数字, "question": "...", "answer": "..."}}]，不要输出其他内容。"""

BATCH_SIZE_CONVERSATIONS = 10
REJECTION_RE = re.compile(
    r"无法回答|无法提供|无法查询|暂?未收录|知识库.{0,12}(没有|查不到)|"
    r"暂未查到|查不到.{0,20}(建议|联系)|"
    r"建议.{0,12}人工|请.{0,12}(转接?|联系|咨询)人工|转接人工"
)
INFORMATIVE_RE = re.compile(
    r"支持|承担|需提供|需要先确认|订单号|提交后|申请售后|"
    r"下单时间|订单状态|收银台|工作日"
)


def normalize_text(text: str) -> str:
    """去空白 + 全角转半角 + 剥句尾标点（归一化判重的键）。"""
    normalized = unicodedata.normalize("NFKC", text)
    no_ws = re.sub(r"\s+", "", normalized)
    return no_ws.rstrip("？?！!。.…~～")


def is_rejection_answer(answer: str) -> bool:
    if INFORMATIVE_RE.search(answer):
        return False
    if REJECTION_RE.search(answer):
        return True
    return len(answer) <= 80 and bool(re.search(r"转人工|人工客服|建工单", answer))


def load_conversation_batches(session: Session, limit: int | None = None):
    """按会话分组读历史消息，(10 会话/批) 产出对话批次。"""
    conv_rows = session.scalars(
        select(Conversation).where(Conversation.status != "已结束").order_by(Conversation.id)
    ).all()
    if limit:
        conv_rows = conv_rows[:limit]
    batches, current = [], []
    for conv in conv_rows:
        msgs = session.scalars(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.id)
        ).all()
        turns = [(m.role, m.content or "") for m in msgs if m.content]
        if not turns:
            continue
        current.append(
            {"conversation_id": conv.id, "messages": [{"role": r, "content": c} for r, c in turns]}
        )
        if len(current) >= BATCH_SIZE_CONVERSATIONS:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


async def extract_batch(session: Session, llm, batch: list[dict], batch_no: str) -> int:
    """一批对话 → LLM 抽 QA → 暂存表（status=extracted）。返回抽出条数。"""
    rendered = []
    for conv in batch:
        rendered.append(f"=== 会话 {conv['conversation_id']} ===")
        for m in conv["messages"]:
            rendered.append(f"[{m['role']}] {m['content']}")
    prompt = ChatPromptTemplate.from_messages(
        [("system", MINER_SYSTEM), ("human", "{dialogues}")]
    ).invoke({"dialogues": "\n".join(rendered)})


    raw = await llm.ainvoke(prompt.to_messages())
    try:
        text = raw.text if not isinstance(raw, list) else json.dumps(raw)
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        qas = json.loads(cleaned)
        if not isinstance(qas, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.warning("mine: LLM 返回非法 JSON，本批跳过（batch=%s）", batch_no)
        return 0
    valid_ids = {c["conversation_id"] for c in batch}
    added = 0
    for qa in qas:
        if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
            continue
        if is_rejection_answer(str(qa["answer"])):
            continue
        conv_id = qa.get("conversation_id")
        source = f"conv:{conv_id}" if conv_id in valid_ids else f"batch:{batch_no}"
        session.add(
            QaExtractionStaging(
                batch_no=batch_no,
                source_ref=source,
                question=qa["question"],
                answer=qa["answer"],
            )
        )
        added += 1
    session.commit()
    return added


def dedup_staging(session: Session, category: str = "对话挖掘") -> list[int]:
    """整体去重：已入库/已待审问题都算重复；新 QA 改投人工审核。"""
    rows = session.scalars(
        select(QaExtractionStaging)
        .where(QaExtractionStaging.status == "extracted")
        .order_by(QaExtractionStaging.id)
    ).all()
    seen: set[str] = {
        normalize_text(q)
        for (q,) in session.execute(select(KnowledgeChunk.questions)).all()
    }
    seen.update(
        normalize_text(q)
        for (q,) in session.execute(select(ReviewQueue.std_question)).all()
    )
    review_rows: list[ReviewQueue] = []
    for row in rows:
        key = normalize_text(row.question)
        if key in seen:
            row.status = "discarded"
        else:
            row.status = "kept"
            seen.add(key)
            review_rows.append(
                ReviewQueue(
                    std_question=row.question,
                    sample_answer=row.answer,
                    source=category,
                )
            )
    session.add_all(review_rows)
    session.flush()
    ids = [row.id for row in review_rows]
    session.commit()
    return ids


async def run_mining(max_batches: int | None = None) -> dict:
    """完整挖知识流程（CLI/定时任务共用）：分批抽 → 过滤去重 → 待审。"""
    from app.config import get_settings
    from app.db import get_session_factory, get_engine  # noqa: F401
    from app.llm import build_chat_model

    factory = get_session_factory()
    settings = get_settings()
    llm = build_chat_model(settings)
    with factory() as session:
        batches = load_conversation_batches(session)
    if max_batches:
        batches = batches[:max_batches]

    extracted = 0
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    for i, batch in enumerate(batches):
        batch_no = f"mine-{stamp}-{i:03d}"
        with factory() as session:
            extracted += await extract_batch(session, llm, batch, batch_no)

    with factory() as session:
        kept_ids = dedup_staging(session)
    return {"batches": len(batches), "extracted": extracted, "kept": len(kept_ids)}


if __name__ == "__main__":
    import asyncio
    import json as _json
    import sys

    _max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(_json.dumps(asyncio.run(run_mining(max_batches=_max)), ensure_ascii=False))
