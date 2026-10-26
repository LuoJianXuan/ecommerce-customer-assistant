"""ch05 主力 Agent 节点：手写 ReAct 循环（图里的核心节点）。

设计要点（文档核实 + 前四章约束）：
- 步进用 astream 聚合：模型一轮输出要么是纯文本（终答，token 经 LangChain 回调
  流给客户端的 messages stream），要么带 tool_calls（聚合后执行回灌进下一步）。
  这样终答真流式、工具步不流式——规避百炼 tools×stream 不同用的老约束。
- 停止条件：无 tool_calls 即收敛；max_steps 保险丝（防失控循环）。
- ch07 起上下文经三层装配引擎组装（app/context_assembly.py）：完整历史留在
  State/checkpointer，调模型用裁过带摘要的精简版；固定前缀打头、每轮可变内容
  （证据/订单注记/消解行/梗概）合成一条背景 HumanMessage 挂用户句后，保护前缀缓存。
- 工具复用 ch02 五件套（build_default_registry 按会话构建）；ReAct 循环内的
  AIMessage(tool_calls)+ToolMessage 配对不裁（峰值已在预算公式中扣除）。
- 知识类的检索证据以编号上下文块随背景条注入，引用角标沿用 ch04 规则。
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.tokens import get_token_counter

AGENT_SYSTEM = """你是{platform_name}电商平台的智能客服「小云」，一个会自主使用工具的 ReAct Agent。

## 工作方式
- 需要实时数据（订单/物流/商品价格库存）就调用对应工具，拿到结果再回答；禁止编造。
- 上下文若已附「知识库检索结果」，优先直接引用它回答，不必再调 query_faq。
- 信息不足以行动（如缺订单号）时，直接向用户追问，不要瞎猜。

## 引用规则（对知识库检索结果）
- 检索结果带编号 [n]，回答中用到的每条事实句末带 [n] 角标；只引真实存在的编号。
- 证据不足以可靠回答时，以「抱歉，这个问题我目前无法可靠回答：」开头说明缺什么，不要编造。

## 禁止承诺（负面清单）
- 不承诺到账时间（只说"1-3 个工作日原路返回，以渠道为准"级别）。
- 不承诺赔付金额/比例，涉及赔付只说"由专员跟进核实"。
- 不承诺物流送达时间；不做证据之外的计算、汇总或推论。

## 风格
简体中文口语化，简洁。
查询类结果（物流轨迹/订单状态/保修售后/退款进度）逐行用「标签：值」呈现，例如：
物流单号：SF13158178019239
当前状态：运输中，已到达杭州转运中心
首尾各加一句自然过渡即可；闲聊与知识问答保持对话感，默认 3 句话内说清。"""

AFTERSALE_JUDGE_NOTE = """退款/售后子流程上下文（确定性流程已就位，你只做终判）：
当前订单：{order_no}｜{product_name}｜{amount} 元｜状态 {status}｜下单时间 {created_at}
你的任务：结合订单状态与「知识库检索结果」中的政策条款，判断这一单能不能退/该怎么办，
给出明确结论与下一步（引用政策来源 [n]）；证据不足以判断时说明缺什么，不要编造。
订单数据以本上下文为准，不要另行编造订单信息。"""


def _last_user_text(state) -> str:
    return str(
        next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    )


def _persist_degradation(deps, conv_id: int, assembled) -> None:
    """层1 降级持久化：锚点只前进（旧值 None 或更小才更新），数据不搬。"""
    if not (assembled.degraded and assembled.new_layer1_from):
        return
    if deps.session_factory is None or not conv_id:
        return
    from app.models import Conversation

    with deps.session_factory() as session:
        conv = session.get(Conversation, conv_id)
        if conv is None:
            return
        old = conv.layer1_from_msg_id
        if old is None or assembled.new_layer1_from > old:
            conv.layer1_from_msg_id = assembled.new_layer1_from
            session.commit()
            logging.getLogger("model_ctx").info(
                "层1 降级 msg_id %s→%s conv=%s", old, assembled.new_layer1_from, conv_id
            )


def _log_model_ctx(conv_id: int, assembled, counter) -> None:
    """model_ctx：每轮调模型前，把实际发出的上下文原样打进日志（验收 grep 目标）。"""
    log = logging.getLogger("model_ctx")
    total = assembled.layer1_tokens + assembled.layer2_tokens + counter(
        str(assembled.messages[0].content or "")
    )
    parts = [
        f"model_ctx conv={conv_id} 窗口轮数={assembled.round_count} "
        f"层1≈{assembled.layer1_tokens}tok 层2≈{assembled.layer2_tokens}tok tokens≈{total} "
        f"降级={assembled.degraded} trim兜底={assembled.trimmed}",
        f"model_ctx 摘要={assembled.summary_text!r}",
    ]
    for m in assembled.messages:
        parts.append(
            f"model_ctx msg {type(m).__name__} {counter(str(m.content or ''))}tok "
            f"head={str(m.content or '')[:80]!r}"
        )
    log.info("\n".join(parts))


def _evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return ""
    lines = [f"[{i}] {m.get('question', '')}：{m.get('answer', '')}" for i, m in enumerate(evidence, 1)]
    return "知识库检索结果（可直接引用，编号供角标使用）：\n" + "\n".join(lines)


async def build_agent_registry(deps, conv_id: int):
    """构建 Agent 工具清单：内置+插件+MCP 每轮现问现拿（react 与 ticket_confirm 共用）。"""
    from app.tools import build_default_registry
    from app.tools.mcp_client import merge_mcp_tools

    registry = build_default_registry(deps.session_factory, conv_id, deps.llm)
    merged = await merge_mcp_tools(registry)
    if merged:
        logging.getLogger("tool_engine").info("MCP 工具合并 %d 个 conv=%s", merged, conv_id)
    return registry


async def run_tool(deps, conv_id: int, registry, tool_call: dict, **kw):
    """经执行引擎调用工具（react 与 ticket_confirm 共用——registry+引擎参数束只此一份）。"""
    from app.config import get_settings as _gs
    from app.tools import engine as _engine

    s = _gs()
    return await _engine.execute(
        tool_call, registry,
        conversation_id=conv_id or None,
        session_factory=deps.session_factory,
        default_timeout=s.tool_timeout_seconds,
        max_retries=s.tool_max_retries,
        **kw,
    )


async def agent_node(deps, state) -> dict:
    """ReAct 循环：ch07 起经三层装配引擎组上下文（完整历史留在 State/checkpointer，
    调模型用裁过带摘要的精简版，两份各走各的）；步进聚合 → 工具回灌 → 终答。"""
    from sqlalchemy import select

    from app.config import get_settings
    from app.context_assembly import assemble_context
    from app.context_budget import budget_from_settings
    from app.context_summary import load_anchors_and_summaries
    from app.tools import build_default_registry
    from app.tools import engine as tool_engine

    settings = get_settings()
    counter = get_token_counter()
    conv_id = state.get("conversation_id") or 0
    registry = await build_agent_registry(deps, conv_id)  # 内置+插件+MCP 每轮现问现拿
    bound = deps.llm.bind_tools(registry.all(), temperature=0)

    # ch07：取会话锚点与注入梗概（测试无库环境视作未启用分层）
    summary_upto = layer1_from = None
    summaries: list[str] = []
    if deps.session_factory is not None and conv_id:
        with deps.session_factory() as session:
            summary_upto, layer1_from, summaries = load_anchors_and_summaries(
                session, conv_id, settings.ctx_summary_inject_segments
            )

    # 每轮可变内容（证据/订单终判注记/消解注记）统一进背景条——挂用户句后，绝不占 system
    variable_blocks: list[str] = []
    evidence_block = _evidence_block(state.get("retrieval_evidence") or [])
    if evidence_block:
        variable_blocks.append(evidence_block)
    order_ctx = state.get("order_context") or None
    if order_ctx:
        variable_blocks.append(AFTERSALE_JUDGE_NOTE.format(
            order_no=order_ctx.get("order_no"),
            product_name=order_ctx.get("product_name"),
            amount=order_ctx.get("amount"),
            status=order_ctx.get("status"),
            created_at=order_ctx.get("created_at"),
        ))
    resolved = str(state.get("resolved_query") or "").strip()
    if resolved == _last_user_text(state):
        resolved = ""  # 透传轮不再重复注入（ch06 既有守卫，ch07 重构时找回）

    assembled = assemble_context(
        system_prompt=AGENT_SYSTEM.format(platform_name=settings.platform_name),
        messages=list(state["messages"]),
        summary_upto=summary_upto,
        layer1_from=layer1_from,
        summaries=summaries,
        evidence_text="\n\n".join(variable_blocks),
        resolved_note=(resolved or None),
        budget=budget_from_settings(settings),
        count_tokens=counter,
        assistant_head_chars=settings.ctx_assistant_head_chars,
    )
    _persist_degradation(deps, conv_id, assembled)
    _log_model_ctx(conv_id, assembled, counter)
    messages = assembled.messages

    new_episode: list = []  # 本回合新增消息（回写 State 由 checkpointer 持久化）
    tool_steps: list[dict] = []  # 本节点实际执行的工具（挂起待确认的不算），API 侧补发徽章帧
    final_reply = ""
    for step in range(1, deps.max_steps + 1):
        gathered: AIMessage | None = None
        text_parts: list[str] = []
        async for chunk in bound.astream(messages):
            gathered = chunk if gathered is None else gathered + chunk
            if chunk.text:
                text_parts.append(chunk.text)
        ai: AIMessage = gathered if gathered is not None else AIMessage(content="")

        if not ai.tool_calls:  # 收敛：终答（token 已在流中经回调送出）
            final_reply = "".join(text_parts) or ai.text
            new_episode.append(ai)
            return {"final_reply": final_reply, "messages": new_episode,
                    "tool_steps": tool_steps,
                    "node_trace": [f"agent:step{step}:final"]}

        # 工具步：统一走执行引擎（权限门/校验/超时重试/分诊/审计全收口）
        messages.append(ai)
        new_episode.append(ai)
        pending_write = None
        for tc in ai.tool_calls:
            out = await run_tool(deps, conv_id, registry, tc)
            if isinstance(out, tool_engine.PendingWrite):
                if pending_write is not None:
                    # spec §7 边界：同回合多个写调用——只挂起第一个,其余权限拒绝回灌,保持配对完整
                    logging.getLogger("tool_engine").warning(
                        "同回合多个写操作,仅首个挂起确认,其余拒绝 conv=%s", conv_id)
                    deny = await run_tool(deps, conv_id, registry, tc, cancelled=True)
                    messages.append(deny.message)
                    new_episode.append(deny.message)
                    continue
                # ch08 方案B：写操作挂起——本轮不执行不产终答，交 ticket_confirm 确认节点
                pending_write = {"call": out.call, "preview": _write_preview(out.call)}
                continue  # 后续只读调用照常执行,AIMessage↔ToolMessage 配对不破
            messages.append(out.message)
            new_episode.append(out.message)
            tool_steps.append({"tool": tc["name"]})
        if pending_write is not None:
            return {"messages": new_episode, "pending_write": pending_write,
                    "tool_steps": tool_steps, "node_trace": ["agent:write_pending"]}

    final_reply = f"这个问题我查了 {deps.max_steps} 步还没能完全确认，建议转人工客服帮您核实，或换个说法我再试试～"
    return {"final_reply": final_reply, "messages": new_episode,
            "tool_steps": tool_steps, "node_trace": ["agent:max_steps"]}


def _write_preview(call: dict) -> dict:
    """写操作挂起时的前端预览数据（当前仅 create_ticket）。"""
    args = call.get("args") or {}
    return {
        "type": "ticket_confirm",
        "call_id": call.get("id", ""),
        "ticket_type": str(args.get("ticket_type") or "咨询"),
        "description": str(args.get("description") or ""),
        "order_id": str(args.get("order_id") or ""),
    }
