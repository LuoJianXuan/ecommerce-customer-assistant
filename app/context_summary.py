"""ch07 会话内摘要域：层2 批次 → 后台异步梗概（主力模型 temperature=0）。

设计要点（spec §4 / 教程文案）：
- 触发看 token 用量（层2 截短口径），不数条数——工具结果不落 messages 表，数条数看不见它涨；
- 任务读**触发时刻的锚点快照**：压缩期间用户继续聊、各轮继续用旧锚点多带一轮原文，
  压完下一轮才收进梗概（宁可多带一轮，不让用户等）；
- 梗概一段段追加（seq 递增）、压完不重写、旧段只作背景不参与合并；注入只取最近 N 段（M4 接线）；
- 失败锚点不动，下次触发重试（批次变大）；同会话同时只跑一个（进程内守门）。
"""

import logging
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import func, select

from app.context_assembly import classify_rounds, layer2_view, split_rounds
from app.models import Conversation, ConversationSummary

logger = logging.getLogger("summary")

SUMMARY_SYSTEM_PROMPT = """你是客服对话的摘要器。把给你的一段客服对话压缩成一段梗概，供后续对话作为背景使用。

规则：
- 只提炼事实与诉求：聊过哪些商品（含型号）、用户报过的订单号/手机号等标识、用户的明确诉求、已经解决与尚未解决的问题；
- 对话里没出现的内容一个字不许编造；寒暄、闲聊、礼貌用语一律不留；
- 输出一段简体中文，几十到两百字，不要分点、不要标题。"""

def load_anchors_and_summaries(session, conversation_id: int, inject_segments: int):
    """取会话双锚点+最近 N 段梗概（react 装配与 resolve_refs 历史视图共用，防两处取数漂移）。

    会话不存在返回 (None, None, [])。
    """
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        return None, None, []
    rows = session.scalars(
        select(ConversationSummary)
        .where(ConversationSummary.conversation_id == conversation_id)
        .order_by(ConversationSummary.seq.desc())
        .limit(inject_segments)
    ).all()
    return (
        conv.summary_upto_msg_id,
        conv.layer1_from_msg_id,
        [r.content for r in reversed(rows)],
    )


_running: set[int] = set()  # 进程内每会话单任务守门


async def summarize_batch(llm, batch_text: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_SYSTEM_PROMPT), ("human", "{dialog}")]
    ).invoke({"dialog": batch_text})
    resp = await llm.ainvoke(prompt.to_messages())
    return str(resp.text if hasattr(resp, "text") else resp).strip()


def layer2_batch(messages: list, summary_upto: int | None, layer1_from: int | None) -> list:
    """当前锚点下属于层2 的轮次（摘要任务的输入范围）。"""
    rounds = split_rounds(messages)
    _l3, l2, _l1 = classify_rounds(rounds, summary_upto, layer1_from)
    return l2


def build_batch_text(rounds: list) -> str:
    """轮次 → 摘要输入转写：用户原话/客服答复全文/工具调用与结果概要行。"""
    lines: list[str] = []
    for r in rounds:
        for m in r.msgs:
            if isinstance(m, HumanMessage):
                lines.append(f"用户：{m.content}")
            elif isinstance(m, AIMessage):
                if str(m.content or "").strip():
                    lines.append(f"客服：{m.content}")
                for tc in m.tool_calls or []:
                    lines.append(f"客服调用了工具 {tc.get('name', '?')}")
            elif isinstance(m, ToolMessage):
                snippet = str(m.content or "")[:200]
                lines.append(f"工具 {getattr(m, 'name', '') or ''}返回：{snippet}")
    return "\n".join(lines)


def should_trigger(messages: list, *, summary_upto, layer1_from, budget, count_tokens, head_chars: int = 80) -> tuple[bool, int]:
    """层2 用量（与装配引擎同口径：截短后的层2 视图 token）是否超层2 预算。"""
    l2 = layer2_batch(messages, summary_upto, layer1_from)
    usage = sum(
        count_tokens(str(m.content or ""))
        for m in layer2_view(l2, head_chars, count_tokens)
    )
    return usage > budget.layer2, usage


def is_running(conversation_id: int) -> bool:
    return conversation_id in _running


async def run_summary_task(
    session_factory, llm, conversation_id: int, *,
    upto: int | None, layer1_from: int | None, messages: list,
    count_tokens, head_chars: int = 80,
) -> None:
    """后台摘要任务：压缩层2 批次 → 追加一段梗概 → 推进 summary_upto 锚点。

    读的是触发时刻的锚点快照（upto/layer1_from），压缩期间的新轮次不受影响。
    """
    if conversation_id in _running:
        logger.info("summary skipped（会话 %s 已有摘要任务在跑）", conversation_id)
        return
    _running.add(conversation_id)
    started = time.perf_counter()
    try:
        batch = layer2_batch(messages, upto, layer1_from)
        batch_text = build_batch_text(batch)
        if not batch_text.strip():
            logger.info("summary skipped（会话 %s 层2 空批）", conversation_id)
            return
        logger.info(
            "summary start conv=%s 批次=%d轮 边界 upto=%s→%s",
            conversation_id, len(batch), upto, layer1_from,
        )
        summary = await summarize_batch(llm, batch_text)
        if not summary:
            raise RuntimeError("摘要输出为空")
        with session_factory() as session:
            seq = (session.execute(
                select(func.max(ConversationSummary.seq)).where(
                    ConversationSummary.conversation_id == conversation_id
                )
            ).scalar() or 0) + 1
            session.add(ConversationSummary(
                conversation_id=conversation_id,
                seq=seq,
                upto_msg_id=layer1_from or 0,
                content=summary,
            ))
            conv = session.get(Conversation, conversation_id)
            conv.summary_upto_msg_id = layer1_from
            session.commit()
        logger.info(
            "summary done 第%d段 conv=%s 边界推进至 %s 耗时 %.1fs 梗概=%s",
            seq, conversation_id, layer1_from, time.perf_counter() - started, summary[:60],
        )
    except Exception:
        logger.exception("summary failed conv=%s 锚点不动（下次触发重试）", conversation_id)
    finally:
        _running.discard(conversation_id)


_FABRICATED_TOKEN_RE = re.compile(r"[A-Z]{2}-[0-9A-Za-z-]{2,}|\d{3,}")  # 编造检测：标识/数字串


def fabrication_check(summary: str, source_text: str) -> list[str]:
    """评估用：摘要里的订单号/数字串必须能在原文找到，找不到即编造。"""
    src = set(_FABRICATED_TOKEN_RE.findall(source_text))
    return [t for t in _FABRICATED_TOKEN_RE.findall(summary) if t not in src]
