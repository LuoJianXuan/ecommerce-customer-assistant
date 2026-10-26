"""SSE 线协议：事件构造与编码。

对话编排不碰线格式，协议变更只改这里。事件负载仍是裸 dict（JSON 直出），
构造器收敛产侧所有 `{"type": ...}` 字面量。
"""

import json


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def delta_event(content: str) -> str:
    return sse_data({"type": "delta", "content": content})


def done_event(assistant_message_id: int | None = None) -> str:
    payload = {"type": "done"}
    if assistant_message_id is not None:
        payload["assistant_message_id"] = assistant_message_id
    return sse_data(payload)


def error_event(message: str) -> str:
    return sse_data({"type": "error", "message": message})


def meta_event(conversation_id: int) -> str:
    """流首帧：告诉页面本次会话的数据库 id，续聊时带回。"""
    return sse_data({"type": "meta", "conversation_id": conversation_id})


def tool_event(tool: str) -> str:
    """工具执行状态帧：页面据此渲染工具轨迹徽章。"""
    return sse_data({"type": "tool", "tool": tool})


def citations_event(items: list[dict]) -> str:
    """证据全集帧：答案角标 [n] → 来源 chunk 的映射（前端浮层数据源）。"""
    return sse_data({"type": "citations", "items": items})
