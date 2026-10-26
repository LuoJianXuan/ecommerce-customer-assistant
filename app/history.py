"""多轮历史裁剪与重建（最简版）：token 预算内从最新往回保留。

- 发送窗口才裁剪，messages 表里存的始终是完整流水。
- 最新的单条消息即使自身超预算也保留，保证对话可继续。
- ch02：历史从 DB 行重建，含工具轮消息（AIMessage(tool_calls)/ToolMessage）。
"""

from collections.abc import Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from app.models import Message
from app.tokens import TokenCounter


def trim_history(
    history: list[BaseMessage], budget: int, count_tokens: TokenCounter
) -> list[BaseMessage]:
    if not history:
        return []

    kept: list[BaseMessage] = []
    total = 0
    for msg in reversed(history):
        cost = count_tokens(str(msg.content))
        if kept and total + cost > budget:
            break
        kept.append(msg)
        total += cost
    kept.reverse()
    return kept


def rows_to_messages(rows: Iterable[Message]) -> list[BaseMessage]:
    """messages 表行 → LangChain 消息（工具轮消息按协议还原）。"""
    out: list[BaseMessage] = []
    for r in rows:
        if r.role == "user":
            out.append(HumanMessage(content=r.content or ""))
        elif r.role == "assistant":
            out.append(
                AIMessage(content=r.content or "", tool_calls=r.tool_calls or [])
            )
        elif r.role == "tool":
            out.append(
                ToolMessage(content=r.content or "", tool_call_id=r.tool_call_id or "")
            )
    return out


def repair_tool_rounds(messages: list[BaseMessage]) -> list[BaseMessage]:
    """修复历史里未闭合的工具回合：AIMessage(tool_calls) 后面没有对应 ToolMessage 时，
    把它降级为普通 assistant 消息（保留正文、剥掉 tool_calls；正文也为空则整条丢弃）。
    防止上游收到协议破损的序列（某些模型会因此幻觉出新的工具调用）。"""
    out: list[BaseMessage] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if isinstance(m, AIMessage) and m.tool_calls:
            j = i + 1
            returned_ids: set[str] = set()
            while j < n and isinstance(messages[j], ToolMessage):
                returned_ids.add(messages[j].tool_call_id)
                j += 1
            requested_ids = {tc["id"] for tc in m.tool_calls}
            if requested_ids <= returned_ids:
                out.append(m)
                out.extend(messages[i + 1 : j])
                i = j
                continue
            if m.content:
                out.append(AIMessage(content=m.content, tool_calls=[]))
            i += 1
            continue
        out.append(m)
        i += 1
    return out


def drop_orphan_tool_prefix(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """窗口头部若以 ToolMessage 开头（其申请单 AIMessage 已被裁掉），丢弃至 HumanMessage 为止，
    避免把无主工具结果发给模型（上游会因 tool_call_id 对不上而报错）。"""
    out = list(messages)
    while out and isinstance(out[0], ToolMessage):
        out.pop(0)
    return out
