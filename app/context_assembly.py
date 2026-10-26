"""ch07 三层装配引擎：State 完整历史 → 发给模型的精简上下文（纯函数）。

分层（按离当前轮的远近）：
- 层3（≤summary_upto）：已被摘要覆盖,不进消息序列,梗概走背景条;
- 层2（summary_upto…layer1_from）：规则截短——用户原话不动、答复留前 N 字、
  工具回合折叠成单条 AIMessage(剥 tool_calls,结果换省略行)——绝不发送
  tool_call 配对破损的序列;
- 层1（≥layer1_from）：原文;超预算先「降级」（最老整轮挪入层2,只前移锚点不搬数据），
  再用 LangChain trim_messages 兜底（保尾部、以 human 开头）。

拼装顺序（固定前缀,服务商前缀缓存友好）：
  System(人设+红线) → 层2截短 → 层1原文(尾部=用户当前句) → 背景条(单条 Human:
  指代消解行+早期梗概≤N段+检索证据)。梗概绝不单独占 system——上游模板会合并所有
  system,把工具定义挤到可变内容之后,前缀缓存整段作废（教程实测）。

锚点口径=messages 表 user 行 id（挂在 HumanMessage.id 上,由 api 落库时回填）;
State 里的标记消息（无 id,如选单回填）永远属于层1。
"""

from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


@dataclass
class Round:
    """一轮对话：一条 user 消息（及其 id 锚点）到下一条 user 消息之前的全部。"""

    key: int | None  # messages 表 user 行 id；无 id（图内标记消息）为 None → 永远层1
    msgs: list[BaseMessage] = field(default_factory=list)


@dataclass
class AssembledContext:
    messages: list[BaseMessage]
    layer1_tokens: int
    layer2_tokens: int
    round_count: int          # 层1 窗口轮数（model_ctx 日志的"窗口条数"）
    degraded: bool            # 本轮是否发生层1 降级
    new_layer1_from: int | None  # 降级后的层1 起点锚点（未降级=None,由调用方决定是否持久化）
    summary_text: str         # 注入的梗概全文（model_ctx 日志用）
    trimmed: bool             # 是否触发 trim_messages 兜底


def _round_key(msg: BaseMessage) -> int | None:
    try:
        return int(msg.id)
    except (TypeError, ValueError):
        return None


def split_rounds(messages: list[BaseMessage]) -> list[Round]:
    rounds: list[Round] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            rounds.append(Round(key=_round_key(m), msgs=[m]))
        else:
            if not rounds:
                rounds.append(Round(key=None))  # 头部孤儿（理论不出现）自成无锚点轮
            rounds[-1].msgs.append(m)
    return [r for r in rounds if r.msgs]


def classify_rounds(
    rounds: list[Round], summary_upto: int | None, layer1_from: int | None
) -> tuple[list[Round], list[Round], list[Round]]:
    """按双锚点切 (层3已摘要, 层2截短, 层1原文)。"""
    l3, l2, l1 = [], [], []
    for r in rounds:
        if r.key is None:
            l1.append(r)  # 图内标记消息永远随层1
        elif summary_upto is not None and r.key <= summary_upto:
            l3.append(r)
        elif layer1_from is not None and r.key < layer1_from:
            l2.append(r)
        else:
            l1.append(r)
    return l3, l2, l1


def _msgs_tokens(msgs: list[BaseMessage], count_tokens) -> int:
    return sum(count_tokens(str(m.content or "")) for m in msgs)


def _fold_nonhuman_run(run: list[BaseMessage], head_chars: int, count_tokens) -> AIMessage:
    """层2：把一段非 user 消息（答复/工具回合）折叠成单条纯文本 AIMessage。"""
    parts: list[str] = []
    for m in run:
        if isinstance(m, AIMessage):
            text = str(m.content or "").strip()
            if text:
                parts.append(text[:head_chars] + ("…" if len(text) > head_chars else ""))
        elif isinstance(m, ToolMessage):
            approx = count_tokens(str(m.content or ""))
            name = getattr(m, "name", None) or "工具"
            parts.append(f"[已调用工具 {name}，结果省略约 {approx} token]")
        else:  # 兜底：其他类型给一行标识
            parts.append(f"[{type(m).__name__} 已省略]")
    content = "\n".join(p for p in parts if p) or "[本轮回合已省略]"
    return AIMessage(content=content, tool_calls=[])


def layer2_view(rounds: list[Round], head_chars: int, count_tokens) -> list[BaseMessage]:
    """层2 视图：每轮 → [Human(原话不动), AIMessage(折叠截短)]，零 ToolMessage 零 tool_calls。"""
    out: list[BaseMessage] = []
    for r in rounds:
        run: list[BaseMessage] = []
        for m in r.msgs:
            if isinstance(m, HumanMessage):
                if run:
                    out.append(_fold_nonhuman_run(run, head_chars, count_tokens))
                    run = []
                out.append(m)
            else:
                run.append(m)
        if run:
            out.append(_fold_nonhuman_run(run, head_chars, count_tokens))
    return out


def _trim_fallback(msgs: list[BaseMessage], max_tokens: int, count_tokens) -> tuple[list[BaseMessage], bool]:
    """层1 兜底：LangChain trim_messages 保尾部（以 human 开头，工具配对不破）。"""
    if _msgs_tokens(msgs, count_tokens) <= max_tokens:
        return msgs, False
    from langchain_core.messages.utils import trim_messages

    trimmed = trim_messages(
        msgs,
        strategy="last",
        max_tokens=max_tokens,
        token_counter=lambda ms: _msgs_tokens(list(ms), count_tokens),
        start_on="human",
        include_system=False,
    )
    trimmed = list(trimmed)
    if not trimmed:
        # 单条消息就超预算时 trim 会清空——保底留「最后一条 user 起」的尾部,对话必须能继续
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                return msgs[i:], True
        return msgs[-1:], True
    return trimmed, True


def assemble_context(
    *,
    system_prompt: str,
    messages: list[BaseMessage],
    summary_upto: int | None,
    layer1_from: int | None,
    summaries: list[str],
    evidence_text: str,
    resolved_note: str | None,
    budget,
    count_tokens,
    assistant_head_chars: int = 80,
) -> AssembledContext:
    rounds = split_rounds(messages)
    _l3, l2_rounds, l1_rounds = classify_rounds(rounds, summary_upto, layer1_from)

    # 层1 降级：超预算把最老整轮挪入层2——只前移锚点，不搬数据；至少保一轮
    degraded = False
    new_layer1_from = layer1_from
    while len(l1_rounds) > 1 and _msgs_tokens([m for r in l1_rounds for m in r.msgs], count_tokens) > budget.layer1:
        l2_rounds.append(l1_rounds.pop(0))
        degraded = True
        if l1_rounds and l1_rounds[0].key is not None:
            new_layer1_from = l1_rounds[0].key

    l2_msgs = layer2_view(l2_rounds, assistant_head_chars, count_tokens)
    l1_msgs_raw = [m for r in l1_rounds for m in r.msgs]
    l1_msgs, trimmed = _trim_fallback(l1_msgs_raw, budget.layer1, count_tokens)

    background = build_background_message(summaries, evidence_text, resolved_note)
    out: list[BaseMessage] = [SystemMessage(content=system_prompt), *l2_msgs, *l1_msgs]
    if background is not None:
        out.append(background)

    return AssembledContext(
        messages=out,
        layer1_tokens=_msgs_tokens(l1_msgs, count_tokens),
        layer2_tokens=_msgs_tokens(l2_msgs, count_tokens),
        round_count=len(l1_rounds),
        degraded=degraded,
        new_layer1_from=new_layer1_from if degraded else None,
        summary_text="\n".join(summaries),
        trimmed=trimmed,
    )


def build_background_message(
    summaries: list[str], evidence_text: str, resolved_note: str | None
) -> HumanMessage | None:
    """背景条：指代消解行 + 早期梗概 + 检索证据，合成一条挂在用户句之后。

    绝不做成 system：上游模板会把所有 system 上提合并，工具定义被挤到可变内容
    之后，前缀缓存整段作废（教程实测教训）。
    """
    blocks: list[str] = []
    if resolved_note:
        blocks.append(f"[指代消解] 用户本轮问题的完整问句：{resolved_note}")
    if summaries:
        blocks.append("[早期对话梗概]\n" + "\n".join(summaries))
    if evidence_text:
        blocks.append(evidence_text)
    if not blocks:
        return None
    return HumanMessage(content="【本轮背景资料】\n" + "\n\n".join(blocks))
