"""ch08 权限策略：我方侧唯一真源——写操作白名单。

规矩（需求原文）：工具分只读/写两类，写操作由执行引擎把门；
MCP Server 的用途声明是对方写的、不可信，能不能调只看这张表，不让模型临场判断。
"""

WRITE_TOOLS = frozenset({"create_ticket"})


def classify_permission(tool_name: str) -> str:
    """工具名 → read/write。默认一律只读；写权限仅来自本表。"""
    return "write" if tool_name in WRITE_TOOLS else "read"


# 写操作业务前置检查：挂起确认**之前**先拦——缺关键信息直接校验拦下回灌，
# 逼模型转向追问用户（「缺问题描述先追问不许瞎编」的引擎侧兜底）
def _ticket_preflight(args: dict) -> str | None:
    if not str(args.get("description", "")).strip():
        return "工单描述为空：建单前必须先向用户问清具体问题（涉及什么商品/订单、出了什么事），不许编造后发起。"
    return None


WRITE_PREFLIGHT = {
    "create_ticket": _ticket_preflight,
}


def preflight_write(tool_name: str, args: dict) -> str | None:
    """写操作前置检查；返回 None=可挂起确认，返回文案=校验拦下。"""
    check = WRITE_PREFLIGHT.get(tool_name)
    return check(args) if check else None
