"""ch05 图骨架：确定性 Workflow 编排（State + 七节点 + 写死分流 + 置信度闸）。

流程（用户需求原文）：指代消解 → 意图识别 → 按意图分流 → 知识检索 → 置信度闸
→ 主力 Agent → 日志记录。分流规则写死在 INTENT_ROUTE：
- 知识类（商品咨询）：强制先 RAG 检索 → 置信度闸 → 证据够才进 Agent
- 业务数据类（物流/订单/其他兜底）：不预检索，直达 Agent 自主调工具
- 退款退货/售后（ch06）：确定性子流程——aftersale_orders（查单+缺单号 interrupt
  弹订单选择器）→ aftersale_policy（Query 扩写+多路检索合并政策条款）→ 主力 Agent
  只做「这一单能不能退」终判
- 投诉：固定安抚话术 + 转人工/建工单两个建议项（零模型、零后端动作）
- 闲聊：固定话术（零模型）
指代消解与意图识别均为 ch06 正式版（app/agent/coref.py / intent.py）。
"""

import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# ---------- 分流规则（写死，八类意图 → 五个路由目标） ----------

INTENT_ROUTE: dict[str, str] = {
    "商品咨询": "knowledge",
    "退款退货": "aftersale",  # ch06：确定性子流程（拿订单+扩写检索政策+agent 终判）
    "物流": "agent",
    "订单": "agent",
    "售后": "aftersale",     # ch06：与退款退货同走一条子流程
    "投诉": "complaint",
    "闲聊": "chat",
    "其他": "agent",  # 兜底类交主力 Agent（有工具、能澄清），不硬塞业务意图
}

ROUTE_TARGET = {
    "knowledge": "retrieve",
    "agent": "agent",
    "complaint": "complaint",
    "chat": "chat",
    "aftersale": "aftersale_orders",
}
DEFAULT_ROUTE = "knowledge"  # 未知意图容错默认：走知识类（有检索+闸保护最稳）

GATE_FALLBACK_REPLY = (
    "抱歉，这个问题我暂时没有查到足够可靠的资料，不敢随便回答以免误导您。"
    "您可以换个说法再问一次，或点击下方联系人工客服核实～"
)

COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验，您的情绪我完全理解，这件事我来帮您跟进。"
    "您可以选择「转人工」直接找人工客服，或选「建工单」让专员按工单处理，两个都可以选，也可以继续和我说～"
)

CHITCHAT_REPLY = (
    "您好呀～我是云优选的智能客服小云。查订单、看物流、退换货、商品咨询这些我都能帮您，"
    "您直接说就行；如果只是想聊两句也没问题，不过专业事我更在行哦～"
)

NO_ORDER_REPLY = (
    "抱歉，我这边暂时查不到您的订单记录，没办法直接判断退换货条件。"
    "您可以稍后再试，或点击下方联系人工客服帮您核实～"
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    intent_confidence: float
    resolved_query: str
    node_trace: Annotated[list[str], operator.add]
    retrieval_evidence: list[dict]
    gate_passed: bool
    final_reply: str
    suggestions: list[dict]
    conversation_id: int
    user_id: str
    order_id: int  # ch06：子流程绑定的订单（会话内点选后复用）
    order_context: dict  # 订单卡片字段（终判注入 agent）
    orders_ready: bool  # ch06：订单绑定是否成功（失败短路直出话术，不进政策检索）
    pending_write: dict  # ch08：写操作挂起（{call, preview}），ticket_confirm 节点消费后清空
    tool_steps: list[dict]  # 对齐轮：本节点实际执行的工具清单（{tool:name}），API 按节点补发徽章帧
    low_confidence_id: int | None  # ch09：闸拦截产生的问题池行，API 负责触发异步归并


@dataclass
class AgentDeps:
    """图的依赖注入束：测试传 fake，生产传真实现。"""

    llm: Any
    retriever: Callable[[str], list[dict]]
    session_factory: Any
    gate_threshold: float = 0.35
    max_steps: int = 5
    # ch06 分流器：意图/消解专用模型（None=退回 llm）；小模型给了才走两级升级
    intent_llm: Any = None
    intent_small_llm: Any = None
    intent_conf_threshold: float = 0.6
    # ch06 子流程：多路检索合并（查询扩写的消费端；None=退回单查询 retriever）
    multi_retriever: Callable[[list[str]], list[dict]] | None = None


# ---------- 节点 ----------

def _last_human_text(state: AgentState) -> str:
    return str(
        next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    )


def current_query(state: AgentState) -> str:
    """本轮判类/检索用问句：优先消解后的完整问句（ch06），没有则退回本轮原句。"""
    return str(state.get("resolved_query") or _last_human_text(state))


async def resolve_refs_node(deps: AgentDeps, state: AgentState) -> dict:
    """指代消解+Query 改写（ch06 正式版；ch07 起历史取层1 滑窗+摘要行）。

    失败降级透传原句；产物写 state.resolved_query，供 classify/检索/子流程共用。
    每轮打 history_ctx（消解/意图共用的历史视图——闲聊/投诉等不进 Agent 的轮次也可见）。
    """
    from app.agent.coref import aresolve

    msgs = list(state["messages"])
    utterance, cut = "", len(msgs)
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            utterance = str(msgs[i].content)
            cut = i
            break

    conv_id = state.get("conversation_id") or 0
    history = msgs[:cut]
    background = ""
    if deps.session_factory is not None and conv_id:
        try:
            from app.config import get_settings
            from app.context_assembly import classify_rounds, split_rounds
            from app.context_summary import load_anchors_and_summaries

            with deps.session_factory() as session:
                summary_upto, layer1_from, segs = load_anchors_and_summaries(
                    session, conv_id, get_settings().ctx_summary_inject_segments
                )
                if segs:
                    background = " / ".join(segs)
                _l3, _l2, l1 = classify_rounds(split_rounds(history), summary_upto, layer1_from)
                history = [m for r in l1 for m in r.msgs]
        except Exception:
            logger.exception("历史视图取数失败，退全量历史")

    result = await aresolve(deps.intent_llm or deps.llm, history, utterance, background=background)
    logging.getLogger("history_ctx").info(
        "history_ctx conv=%s 摘要行=%r 滑窗=%d条\n%s",
        conv_id, background[:80], len(history),
        "\n".join(
            f"history_ctx {type(m).__name__} {str(m.content or '')[:60]!r}" for m in history
        ),
    )
    return {"resolved_query": result.resolved_query, "node_trace": ["resolve_refs"]}


async def classify_intent(deps: AgentDeps, state: AgentState) -> dict:
    """意图识别正式版（ch06）：四件套 prompt + 重试容错 + 「其他」兜底 + 可选两级降级。

    判据用消解后的完整问句（state.resolved_query，M2 起有值），没有则退回本轮原句。
    """
    from app.agent.intent import aclassify

    query = current_query(state)
    result = await aclassify(
        deps.intent_llm or deps.llm,
        query,
        small_llm=deps.intent_small_llm,
        threshold=deps.intent_conf_threshold,
    )
    if result.escalated:  # spec §2：是否升级要记入日志/trace
        logger.info("[intent] 低置信升级大模型重判 conv=%s q=%.60s -> %s(%.2f)",
                    state.get("conversation_id"), query, result.intent, result.confidence)
    if result.parse_failed:
        logger.warning("[intent] JSON 两次解析失败降级「其他」conv=%s q=%.60s",
                       state.get("conversation_id"), query)
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "node_trace": ["classify"],
    }


def retrieve_node(deps: AgentDeps, state: AgentState) -> dict:
    """知识类强制检索（复用 ch04 hybrid_rerank 检索器，经 deps.retriever 注入）。

    ch06 起用消解后的完整问句检索（resolved_query），带指代的问题也能召回正确知识。
    """
    query = current_query(state)
    evidence = deps.retriever(query) or []
    from app.knowledge.flywheel import save_retrieval_snapshot

    save_retrieval_snapshot(
        deps.session_factory, state.get("conversation_id"), query, evidence
    )
    return {"retrieval_evidence": evidence, "node_trace": ["retrieve"]}


def evidence_confidence(evidence: list[dict]) -> dict:
    """用精排 Top1、有效证据数和 Top1/Top2 分差计算证据置信度。"""
    scores = [max(float(item.get("score") or 0), 0.0) for item in evidence]
    top1 = scores[0] if scores else 0.0
    valid_count = sum(score > 0 for score in scores)
    gap = max(top1 - scores[1], 0.0) if len(scores) > 1 else 0.0
    confidence = top1
    if valid_count < 3:
        confidence -= 0.05
    if gap < 0.05:
        confidence -= 0.05
    return {
        "confidence": round(max(confidence, 0.0), 4),
        "top1": round(top1, 4),
        "valid_count": valid_count,
        "gap": round(gap, 4),
    }


def confidence_gate_node(deps: AgentDeps, state: AgentState) -> dict:
    """置信度闸：检索后、进 Agent 前。证据弱 → 兜底话术不进 Agent，问题入低置信池。"""
    evidence = state.get("retrieval_evidence") or []
    signals = evidence_confidence(evidence)
    passed = signals["confidence"] >= deps.gate_threshold
    if passed:
        return {"gate_passed": True, "node_trace": ["gate:pass"]}
    reason = (
        f"evidence_confidence={signals['confidence']:.4f}; "
        f"top1={signals['top1']:.4f}; valid_count={signals['valid_count']}; "
        f"gap={signals['gap']:.4f}; threshold={deps.gate_threshold:.4f}"
    )
    row_id = _record_low_confidence(deps, state, reason)
    return {
        "gate_passed": False,
        "final_reply": GATE_FALLBACK_REPLY,
        "low_confidence_id": row_id,
        "node_trace": ["gate:block"],
    }


def _record_low_confidence(deps: AgentDeps, state: AgentState, reason: str) -> int | None:
    if deps.session_factory is None:
        return None
    try:
        from app.knowledge.flywheel import evidence_snapshot, record_low_confidence

        last = _last_human_text(state)
        return record_low_confidence(
            deps.session_factory,
            conversation_id=state.get("conversation_id"),
            raw_question=str(last),
            source="retrieval_low_conf",
            reason=reason,
            retrieval_snapshot=evidence_snapshot(state.get("retrieval_evidence") or []),
        )
    except Exception:
        logger.exception("置信度闸落低置信池失败")
        return None


def complaint_node(deps: AgentDeps, state: AgentState) -> dict:
    """投诉出口：固定安抚话术 + 转人工/建工单两个建议项（零模型、零后端动作）。"""
    last = str(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    )
    return {
        "final_reply": COMPLAINT_REPLY,
        "suggestions": [
            {"label": "转人工", "action": "transfer_human"},
            {
                "label": "建工单",
                "action": "create_ticket",
                "payload": {"description": f"用户投诉：{last}", "ticket_type": "投诉"},
            },
        ],
        "node_trace": ["complaint"],
    }


def chitchat_node(deps: AgentDeps, state: AgentState) -> dict:
    """闲聊出口：固定话术（零模型调用）。"""
    return {"final_reply": CHITCHAT_REPLY, "node_trace": ["chat"]}


# ---------- 退款/售后确定性子流程（ch06） ----------

async def aftersale_orders_node(deps: AgentDeps, state: AgentState) -> dict:
    """子流程第一步（确定性，无 LLM）：查订单（懒播种）→ 绑定订单。

    槽位规则（需求原文）：缺订单号不让模型猜，执行阶段弹订单选择器让用户点选回填。
    会话内已选过（state.order_id）直接复用；仅一单自动绑定；多单未选 → interrupt
    暂停弹选择器，前端点选后以 Command(resume={"order_id": ...}) 接续。
    恢复时节点从头重跑——查单幂等，无害。
    """
    from langgraph.types import interrupt

    from app.agent.orders import (
        extract_order_refs,
        get_or_seed_user_orders,
        match_order_ref,
        order_summary,
    )
    from app.models import Order

    user_id = state.get("user_id") or ""
    order: Order | None = None

    chosen_id = state.get("order_id")
    if chosen_id:
        with deps.session_factory() as session:
            order = session.get(Order, chosen_id)
        if order is not None and order.user_id != user_id:
            order = None  # 归属不符视为未选

    if order is None:
        with deps.session_factory() as session:
            orders = get_or_seed_user_orders(session, user_id)
        if not orders:
            return {
                "orders_ready": False,
                "final_reply": NO_ORDER_REPLY,
                "node_trace": ["aftersale_orders:empty"],
            }
        # 槽位提取（修复轮）：消费消解问句+近几轮里用户点名的订单号，命中直接绑定
        ref_text = " ".join(
            [str(state.get("resolved_query") or "")]
            + [
                str(m.content)
                for m in list(state["messages"])[-8:]
                if isinstance(m, HumanMessage)
            ]
        )
        refs = extract_order_refs(ref_text)
        for ref in refs:
            hit = match_order_ref(orders, ref)
            if hit is not None:
                order = hit
                break
        if order is None:
            if len(orders) == 1 and not refs:
                order = orders[0]
            else:
                note = (
                    f"没有找到订单 {' / '.join(refs)} 哦，您名下有这些订单，点选要处理的一单："
                    if refs
                    else "请点击选择要处理的订单，选好我就接着判断这一单～"
                )
                picked = interrupt(
                    {"type": "order_picker", "note": note,
                     "orders": [order_summary(o) for o in orders]}
                )
                pid = picked.get("order_id") if isinstance(picked, dict) else None
                with deps.session_factory() as session:
                    order = session.get(Order, pid) if pid else None
                if order is None or order.user_id != user_id:
                    return {
                        "orders_ready": False,
                        "final_reply": "订单选择出了点问题，请重新说一下您的退换货需求～",
                        "node_trace": ["aftersale_orders:invalid_pick"],
                    }

    ctx = order_summary(order)
    marker = HumanMessage(content=f"（已选择订单：{ctx['order_no']} {ctx['product_name']}）")
    return {
        "orders_ready": True,
        "order_id": order.id,
        "order_context": ctx,
        "messages": [marker],
        "node_trace": ["aftersale_orders"],
    }


async def aftersale_policy_node(deps: AgentDeps, state: AgentState) -> dict:
    """子流程第二步（确定性编排 + LLM 扩写）：Query 扩写 → 多路检索合并 → 政策证据。

    收尾建议（spec 定稿）：退款退货 → 退款表单入口；售后 → 复用 ch05 工单入口。
    """
    from app.agent.query_expand import aexpand

    query = current_query(state)
    queries = await aexpand(deps.intent_llm or deps.llm, query)
    if deps.multi_retriever is not None:
        evidence = deps.multi_retriever(queries) or []
    else:
        evidence = deps.retriever(queries[0]) or []

    from app.knowledge.flywheel import save_retrieval_snapshot

    save_retrieval_snapshot(
        deps.session_factory, state.get("conversation_id"), query, evidence
    )

    ctx = state.get("order_context") or {}
    suggestions: list[dict] = []
    if ctx.get("id") is not None:
        if state.get("intent") == "退款退货":
            suggestions = [{
                "label": "提交退款申请",
                "action": "refund_form",
                "payload": {"order_id": ctx["id"], "order_no": ctx.get("order_no")},
            }]
        elif state.get("intent") == "售后":
            suggestions = [{
                "label": "建工单",
                "action": "create_ticket",
                "payload": {
                    "description": f"售后诉求（订单 {ctx.get('order_no')}）：{query}",
                    "ticket_type": "售后",
                },
            }]
    return {
        "retrieval_evidence": evidence,
        "suggestions": suggestions,
        "node_trace": ["aftersale_policy"],
    }


async def ticket_confirm_node(deps: AgentDeps, state: AgentState) -> dict:
    """ch08 写操作确认节点（确定性零模型，方案B）：interrupt 推预览，resume 确认/取消。

    确认 → 引擎真执行（审计「成功」）；取消 → 引擎记权限拒绝。两种结果都补上与
    call_id 配对的 ToolMessage 回灌，agent 重入后模型据此收尾。节点重跑零成本。
    """
    from langgraph.types import interrupt

    from app.agent.react import build_agent_registry, run_tool

    pending = state.get("pending_write") or {}
    call = pending.get("call") or {}
    preview = pending.get("preview") or {}
    answer = interrupt(preview)
    confirmed = bool(isinstance(answer, dict) and answer.get("confirmed"))

    conv_id = state.get("conversation_id") or 0
    registry = await build_agent_registry(deps, conv_id)
    out = await run_tool(deps, conv_id, registry, call, confirmed=confirmed, cancelled=not confirmed)
    return {
        "messages": [out.message],
        "pending_write": None,
        # 取消轮是权限拒绝（没真调），不挂徽章；仅确认执行才记
        "tool_steps": [{"tool": call.get("name", "")}] if confirmed else [],
        "node_trace": ["ticket_confirm:" + ("confirmed" if confirmed else "cancelled")],
    }


def logger_node(deps: AgentDeps, state: AgentState) -> dict:
    """日志节点：整条节点轨迹落日志（验收 1 的"日志里看到强制检索被走到"）。"""
    trace = " → ".join(state.get("node_trace") or [])
    logger.info("[agent-graph] conv=%s intent=%s trace=%s",
                state.get("conversation_id"), state.get("intent"), trace)
    return {}


def route_by_intent(state: AgentState) -> str:
    return ROUTE_TARGET[INTENT_ROUTE.get(state.get("intent"), DEFAULT_ROUTE)]


def route_after_orders(state: AgentState) -> str:
    """订单绑定失败（无订单/点选无效）短路直出话术，不进政策检索。"""
    return "aftersale_policy" if state.get("orders_ready") else "logger"


def route_after_agent(state: AgentState) -> str:
    """ch08：Agent 挂起写操作 → 确认节点；否则收尾。"""
    return "ticket_confirm" if state.get("pending_write") else "logger"


def route_by_gate(state: AgentState) -> str:
    return "agent" if state.get("gate_passed") else "logger"


# ---------- 建图 ----------

def build_graph(deps: AgentDeps):
    """组装图。agent 节点由 react 模块提供（M3）；checkpointer 由调用方传入。"""
    from app.agent.react import agent_node

    async def _agent_node(s):
        return await agent_node(deps, s)

    async def _resolve_node(s):
        return await resolve_refs_node(deps, s)

    async def _classify_node(s):
        return await classify_intent(deps, s)

    async def _aftersale_orders(s):
        return await aftersale_orders_node(deps, s)

    async def _aftersale_policy(s):
        return await aftersale_policy_node(deps, s)

    async def _ticket_confirm(s):
        return await ticket_confirm_node(deps, s)

    builder = StateGraph(AgentState)
    builder.add_node("resolve_refs", _resolve_node)
    builder.add_node("classify", _classify_node)
    builder.add_node("aftersale_orders", _aftersale_orders)
    builder.add_node("aftersale_policy", _aftersale_policy)
    builder.add_node("ticket_confirm", _ticket_confirm)
    builder.add_node("retrieve", lambda s: retrieve_node(deps, s))
    builder.add_node("gate", lambda s: confidence_gate_node(deps, s))
    builder.add_node("agent", _agent_node)
    builder.add_node("complaint", lambda s: complaint_node(deps, s))
    builder.add_node("chat", lambda s: chitchat_node(deps, s))
    builder.add_node("logger", lambda s: logger_node(deps, s))

    builder.add_edge(START, "resolve_refs")
    builder.add_edge("resolve_refs", "classify")
    builder.add_conditional_edges(
        "classify", route_by_intent,
        {
            "retrieve": "retrieve",
            "agent": "agent",
            "complaint": "complaint",
            "chat": "chat",
            "aftersale_orders": "aftersale_orders",
        },
    )
    builder.add_conditional_edges(
        "aftersale_orders", route_after_orders,
        {"aftersale_policy": "aftersale_policy", "logger": "logger"},
    )
    builder.add_edge("aftersale_policy", "agent")
    builder.add_edge("retrieve", "gate")
    builder.add_conditional_edges("gate", route_by_gate, {"agent": "agent", "logger": "logger"})
    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"ticket_confirm": "ticket_confirm", "logger": "logger"},
    )
    builder.add_edge("ticket_confirm", "agent")  # 确认/取消后回 Agent 收尾（零重放）
    builder.add_edge("complaint", "logger")
    builder.add_edge("chat", "logger")
    builder.add_edge("logger", END)
    return builder


# ---------- checkpointer（官方 SqliteSaver 的异步版，落 .checkpoints.sqlite） ----------

_DB_PATH = None
_checkpointer = None
_conn = None


async def get_checkpointer(db_path: str | None = None):
    """AsyncSqliteSaver 单例（服务端 astream 必须 async 版；.checkpoints.sqlite，
    setup() 建表一次；DBeaver 可开该文件看三张表）。"""
    global _DB_PATH, _checkpointer, _conn
    if _checkpointer is None:
        from pathlib import Path

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        _DB_PATH = db_path or str(Path(__file__).resolve().parent.parent.parent / ".checkpoints.sqlite")
        _conn = await aiosqlite.connect(_DB_PATH).__aenter__()
        _checkpointer = AsyncSqliteSaver(_conn)
        await _checkpointer.setup()
    return _checkpointer


async def close_checkpointer() -> None:
    """关闭底层 aiosqlite 连接（其工作线程非守护，脚本不关会在退出时挂死）。"""
    global _checkpointer, _conn
    if _conn is not None:
        await _conn.close()
        _conn = None
        _checkpointer = None
