"""ch05/ch06 Agent API：/api/agent/chat（图引擎 SSE，含 interrupt 续跑）、
/api/agent/ticket（建工单）、/api/agent/refund（退款表单提交）。

SSE 契约：兼容旧帧（meta/tool[agent 工具步]/delta/citations/done）+ 新帧
- {type:"node", name}：节点轨迹
- {type:"suggestions", items}：建议项（投诉出口、退款表单/工单入口）
- {type:"order_picker", orders}：ch06 订单选择器——子流程 interrupt 暂停时下发，
  前端渲染可点选订单卡片，点选后以 {resume:{order_id}} 再调本接口接续
请求：message（正常消息）与 resume（中断点选回填）二选一。
陈旧帧防护：suggestions/citations 只在本轮确实跑到会产出它们的节点时才发
（checkpointer 里上一轮的残留不再重发——ch05 遗留小 bug 顺手修掉）。
"""

import asyncio
import json
import logging
from contextlib import nullcontext

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, Session

from app.agent.graph import AgentDeps, build_graph, get_checkpointer
from app.config import Settings, get_settings
from app.db import get_session_factory
from app.llm import build_chat_model, build_intent_model
from app.models import Conversation, Message
from app.schemas import ChatRequest
from app.sse import citations_event, delta_event, done_event, error_event, meta_event, tool_event

router = APIRouter(prefix="/api/agent", tags=["agent"])


def schedule_low_confidence(row_id: int) -> None:
    from app.knowledge.flywheel import schedule_low_confidence as schedule

    schedule(row_id)


class AgentChatRequest(BaseModel):
    """ch06：message 与 resume 二选一——resume 携带订单选择器等中断的点选结果。"""

    conversation_id: int | None = None
    user_id: str = Field(default="web-anon", min_length=1, max_length=64)
    message: str = ""
    resume: dict | None = None


class FeedbackRequest(BaseModel):
    conversation_id: int
    message_id: int | None = None
    helpful: bool


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest) -> dict:
    """👎 将对应轮次的用户原话和最近检索快照送进数据飞轮。"""
    from app.knowledge.flywheel import latest_retrieval_snapshot, record_low_confidence

    if req.helpful:
        return {"status": "recorded"}

    factory = get_session_factory()
    with factory() as session:
        conv = session.get(Conversation, req.conversation_id)
        if conv is None:
            raise HTTPException(404, f"会话不存在: {req.conversation_id}")
        if req.message_id is not None:
            assistant = session.get(Message, req.message_id)
            if assistant is None:
                raise HTTPException(422, "message_id 对应的客服回复不存在")
            if assistant.conversation_id != req.conversation_id or assistant.role != "assistant":
                raise HTTPException(422, "message_id 不是该会话的客服回复")
        else:
            assistant = session.scalar(
                select(Message)
                .where(Message.conversation_id == req.conversation_id, Message.role == "assistant")
                .order_by(Message.id.desc())
                .limit(1)
            )
        if assistant is None:
            raise HTTPException(422, "该会话还没有可评价的客服回复")
        user_message = session.scalar(
            select(Message)
            .where(
                Message.conversation_id == req.conversation_id,
                Message.role == "user",
                Message.id < assistant.id,
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
        if user_message is None:
            raise HTTPException(422, "没有找到该回复对应的用户问题")
        snapshot = latest_retrieval_snapshot(
            session,
            req.conversation_id,
            before=assistant.created_at,
            after=user_message.created_at,
        )
        raw_question = str(user_message.content or "")

    row_id = record_low_confidence(
        factory,
        conversation_id=req.conversation_id,
        raw_question=raw_question,
        source="user_feedback",
        reason="用户反馈未解决",
        retrieval_snapshot=snapshot,
    )
    schedule_low_confidence(row_id)
    return {
        "status": "queued",
        "low_confidence_id": row_id,
        "raw_question": raw_question,
    }


def _production_deps(settings: Settings) -> AgentDeps:
    from app.knowledge.retrieval import retrieve, retrieve_multi_query

    factory = get_session_factory()
    return AgentDeps(
        llm=build_chat_model(settings),
        retriever=lambda q: retrieve(
            factory, q, top_k=settings.rerank_top_k, strategy="hybrid_rerank"
        ),
        multi_retriever=lambda queries: retrieve_multi_query(factory, queries, final_k=5),
        session_factory=factory,
        gate_threshold=settings.evidence_conf_threshold,
        max_steps=settings.agent_max_steps,
        intent_llm=build_intent_model(settings),
        intent_small_llm=(
            build_intent_model(settings, small=True)
            if settings.intent_escalation_enabled
            else None
        ),
        intent_conf_threshold=settings.intent_confidence_threshold,
    )


def _langfuse_handler(settings) -> object | None:
    """ch09 M0：每请求一个 CallbackHandler；未启用/未配 key/导入失败 → None，聊天零影响。"""
    if not settings.langfuse_enabled or not settings.langfuse_public_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        logging.getLogger("langfuse").exception("CallbackHandler 创建失败，本轮不追踪")
        return None


def _langfuse_patch_metadata(meta: dict) -> None:
    """ch09 M1：在当前 trace 上下文内补 metadata+tag（意图在 classify 节点后才Known）。
    intent 同时打 tag——events_only 模式下 UI 的 Trace Tags 列可直接筛/分组做成本归因。
    进出即生效的 with 形态；失败只记日志，绝不影响回复。"""
    try:
        from langfuse import propagate_attributes

        intent = str(meta.get("intent") or "")
        tags = [f"intent:{intent}"] if intent else []
        with propagate_attributes(metadata=meta, tags=tags):
            pass
    except Exception:
        logging.getLogger("langfuse").warning("trace metadata 补写失败", exc_info=True)


@router.post("/chat")
async def agent_chat(req: AgentChatRequest) -> None:
    from fastapi.responses import StreamingResponse

    if req.resume is None and not req.message.strip():
        raise HTTPException(422, "message 与 resume 必须提供其一")

    settings = get_settings()
    factory: sessionmaker[Session] = get_session_factory()

    async def stream():
        try:
            is_resume = req.resume is not None
            user_row_id: int | None = None
            user_created_at = None
            with factory() as session:
                conv = session.get(Conversation, req.conversation_id) if req.conversation_id else None
                if conv is None:
                    conv = Conversation(user_id=req.user_id)
                    session.add(conv)
                    session.commit()
                conv_id = conv.id
                if is_resume:
                    resume_note = (
                        f"（点选订单回填：{req.resume.get('order_id')}）"
                        if req.resume.get("order_id")
                        else f"（工单确认：{'已确认' if req.resume.get('confirmed') else '已取消'}）"
                    )
                    session.add(Message(conversation_id=conv_id, role="user", content=resume_note))
                    session.commit()
                else:
                    # ch07：user 行 id 挂到消息 id 上——三层装配的锚点口径
                    user_msg = Message(conversation_id=conv_id, role="user", content=req.message)
                    session.add(user_msg)
                    session.commit()
                    user_row_id = user_msg.id
                    user_created_at = user_msg.created_at

            yield meta_event(conv_id)

            checkpointer = await get_checkpointer()
            graph = build_graph(_production_deps(settings)).compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": str(conv_id)}}
            if is_resume:
                from langgraph.types import Command

                graph_input = Command(resume=req.resume)
            else:
                graph_input = {
                    "messages": [HumanMessage(content=req.message, id=str(user_row_id))],
                    "conversation_id": conv_id,
                    "user_id": req.user_id,
                    "low_confidence_id": None,
                }

            # ch09 M0：Langfuse 追踪——每请求一个 handler，session=会话、user=访客
            handler = _langfuse_handler(settings)
            lf_root_ctx = nullcontext(None)
            lf_ctx = nullcontext()
            if handler is not None:
                from langfuse import get_client, propagate_attributes

                lf_root_ctx = get_client().start_as_current_observation(name="agent-chat")
                lf_ctx = propagate_attributes(
                    trace_name="agent-chat", session_id=str(conv_id), user_id=req.user_id
                )

            interrupt_payload = None
            nodes_seen: set[str] = set()
            intent_meta: dict = {"entry": "agent"}  # classify 帧到达时补意图维度
            with lf_root_ctx:
                with lf_ctx:
                    if handler is not None:
                        config = {**config, "callbacks": [handler]}
                    async for payload in graph.astream(graph_input, config, stream_mode="updates"):
                        for node_name, diff in (payload or {}).items():
                            if node_name == "classify" and isinstance(diff, dict):
                                intent_meta.update({
                                    "intent": diff.get("intent"),
                                    "intent_confidence": diff.get("intent_confidence"),
                                })
                            if node_name == "__interrupt__":
                                intr = diff[0] if isinstance(diff, tuple) and diff else None
                                interrupt_payload = getattr(intr, "value", None) if intr is not None else None
                                continue
                            if node_name == "__end__" or diff is None:
                                continue
                            nodes_seen.add(node_name)
                            yield f'data: {json.dumps({"type": "node", "name": node_name}, ensure_ascii=False)}\n\n'
                            # 对齐轮：节点带回的执行工具清单 → 徽章帧（react/ticket_confirm 收集）
                            for tstep in diff.get("tool_steps") or []:
                                if tstep.get("tool"):
                                    yield tool_event(tstep["tool"])

                    state = (await graph.aget_state(config)).values or {}
                    # 意图在图运行后才可用；此时显式 agent-chat 根 span 仍为 current，
                    # propagate_attributes 会把 metadata/tag 写到根 span，而不是空上下文。
                    if handler is not None:
                        if not intent_meta.get("intent"):
                            intent_meta["intent"] = state.get("intent") or "未知"
                        _langfuse_patch_metadata(intent_meta)

            gate_row_id = state.get("low_confidence_id")
            if gate_row_id:
                schedule_low_confidence(gate_row_id)

            # ch06 修复轮：消解提示条——消解版与原句不同才透出（教程式「🔍 指代消解」）。
            # 放在 interrupt 早退之前：弹选择器的那一轮正是消解发挥作用的轮次。
            # 原句选取要跳过子流程追加的「（已选择订单…）」标记消息，否则绑定轮会拿标记当原句。
            resolved = str(state.get("resolved_query") or "").strip()
            last_user = str(
                next(
                    (m.content for m in reversed(state.get("messages") or [])
                     if isinstance(m, HumanMessage)
                     and not str(m.content).startswith("（已选择订单")),
                    "",
                )
            ).strip()
            if not is_resume and resolved and last_user and resolved != last_user:
                yield f'data: {json.dumps({"type": "resolved", "raw": last_user, "resolved": resolved}, ensure_ascii=False)}\n\n'

            if interrupt_payload is not None:
                # 子流程在订单选择器处暂停：下发卡片数据，本轮无终答；点选后 resume 接续
                yield f'data: {json.dumps(interrupt_payload, ensure_ascii=False)}\n\n'
                yield done_event()
                return

            # 终答伪流式（agent 节点聚合后输出；工具回合的思考文本/工具 JSON 不泄漏进流）
            final = str(state.get("final_reply") or "")
            if final:
                for i in range(0, len(final), 3):
                    yield delta_event(final[i : i + 3])
                    await asyncio.sleep(0.02)

            evidence = state.get("retrieval_evidence") or []
            if evidence and ({"retrieve", "aftersale_policy"} & nodes_seen):
                yield citations_event([
                    {"n": i, "chunk_id": m.get("chunk_id"), "section_path": m.get("section_path"),
                     "question": m.get("question"), "answer": m.get("answer")}
                    for i, m in enumerate(evidence, 1)
                ])
            suggestions = state.get("suggestions") or []
            if suggestions and ({"complaint", "aftersale_policy"} & nodes_seen):
                yield f'data: {json.dumps({"type": "suggestions", "items": suggestions}, ensure_ascii=False)}\n\n'

            assistant_message_id = None
            assistant_created_at = None
            if final:
                with factory() as session:
                    assistant = Message(conversation_id=conv_id, role="assistant", content=final)
                    session.add(assistant)
                    session.commit()
                    assistant_message_id = assistant.id
                    assistant_created_at = assistant.created_at

                from app.knowledge.flywheel import (
                    evidence_snapshot,
                    latest_retrieval_snapshot,
                    record_low_confidence,
                )
                from app.prompts import REFUSAL_PREFIX

                if final.strip().startswith(REFUSAL_PREFIX):
                    current_evidence = (
                        evidence if ({"retrieve", "aftersale_policy"} & nodes_seen) else []
                    )
                    if not current_evidence and "agent" in nodes_seen:
                        with factory() as session:
                            current_evidence = latest_retrieval_snapshot(
                                session,
                                conv_id,
                                before=assistant_created_at,
                                after=user_created_at,
                            ) or []
                    row_id = record_low_confidence(
                        factory,
                        conversation_id=conv_id,
                        raw_question=last_user,
                        source="self_check",
                        reason=final,
                        retrieval_snapshot=evidence_snapshot(current_evidence),
                    )
                    schedule_low_confidence(row_id)

            # ch07：层2 超预算 → 后台摘要任务（锚点快照；不阻塞本轮回复，压完下一轮生效）
            if interrupt_payload is None:
                try:
                    from app.context_budget import budget_from_settings
                    from app.context_summary import run_summary_task, should_trigger
                    from app.llm import build_summary_model
                    from app.models import Conversation as ConvRow
                    from app.tokens import get_token_counter

                    with factory() as session:
                        conv_row = session.get(ConvRow, conv_id)
                        snap_upto = conv_row.summary_upto_msg_id if conv_row else None
                        snap_layer1 = conv_row.layer1_from_msg_id if conv_row else None
                    counter = get_token_counter()
                    budget = budget_from_settings(settings)
                    trigger, usage = should_trigger(
                        list(state.get("messages") or []),
                        summary_upto=snap_upto, layer1_from=snap_layer1,
                        budget=budget, count_tokens=counter,
                    )
                    if trigger:
                        logging.getLogger("summary").info(
                            "summary trigger conv=%s 层2约 %d token > 预算 %d",
                            conv_id, usage, budget.layer2,
                        )
                        asyncio.create_task(run_summary_task(
                            factory, build_summary_model(settings), conv_id,
                            upto=snap_upto, layer1_from=snap_layer1,
                            messages=list(state.get("messages") or []),
                            count_tokens=counter,
                        ))
                except Exception:
                    logging.getLogger("summary").exception("摘要触发检查失败（不影响回复）")

            yield done_event(assistant_message_id)
        except Exception as exc:
            yield error_event(f"{type(exc).__name__}: {exc}")

    from fastapi.responses import StreamingResponse as SR

    return SR(stream(), media_type="text/event-stream",
              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class TicketRequest(BaseModel):
    conversation_id: int
    description: str = Field(min_length=1, max_length=2000)
    ticket_type: str = Field(default="投诉", pattern="^(售后|投诉|咨询)$")


@router.post("/ticket")
def create_ticket(req: TicketRequest) -> dict:
    """仅当用户点「建工单」按钮时才调用（后端从不自动建单）。"""
    from app.models import Ticket
    from app.tools.tickets import gen_ticket_no

    factory = get_session_factory()
    with factory() as session:
        if session.get(Conversation, req.conversation_id) is None:
            raise HTTPException(404, f"会话不存在: {req.conversation_id}")
        ticket_no = gen_ticket_no()
        session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=req.conversation_id,
                description=req.description,
                ticket_type=req.ticket_type,
            )
        )
        session.commit()
    return {"ticket_no": ticket_no, "status": "待处理"}


class RefundForm(BaseModel):
    """退款表单提交（ch06）：原因从固定类目下拉选，前端不传自由文本。"""

    conversation_id: int
    user_id: str = Field(min_length=1, max_length=64)
    order_id: int
    reason: str


@router.post("/refund")
def submit_refund(req: RefundForm) -> dict:
    """仅当用户在退款表单点「提交」时才调用；类目与订单归属后端再校验一遍。"""
    from app.agent.orders import REFUND_REASONS, create_refund

    if req.reason not in REFUND_REASONS:
        raise HTTPException(422, f"退款原因必须是固定类目之一: {REFUND_REASONS}")
    factory = get_session_factory()
    with factory() as session:
        if session.get(Conversation, req.conversation_id) is None:
            raise HTTPException(404, f"会话不存在: {req.conversation_id}")
        try:
            refund_no = create_refund(session, req.order_id, req.user_id, req.reason)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"refund_no": refund_no, "status": "待审核"}
