"""ch08 执行引擎（TDD）：重试策略/写操作不重试/三分诊/格式化/审计五状态。

引擎内存语义用 fake 工具+内存 registry；审计用真 MySQL（require_db）。
"""

import asyncio
import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from app.tools.base import ToolRegistry
from app.tools.engine import PendingWrite, ToolExecution, execute, format_tool_result


@tool
def ok_tool(keyword: str) -> str:
    """正常工具。"""
    return f"结果:{keyword}"


@tool
def dict_tool(order_id: str) -> dict:
    """返回 dict,带内部字段与枚举码。"""
    return {"status": "shipped", "recall_score": 0.33, "carrier": "顺丰", "name": "云优选"}


@tool
def slow_tool(order_id: str) -> str:
    """永远超时。"""
    await_count["n"] += 1
    import time

    time.sleep(0.2)
    return "never"


@tool
def flaky_tool(order_id: str) -> str:
    """网络抖动两次后成功。"""
    calls["n"] += 1
    if calls["n"] <= 2:
        raise ConnectionError("上游抖动")
    return "恢复成功"


@tool
def broken_tool(order_id: str) -> str:
    """业务异常。"""
    raise ValueError("业务炸了")


@tool
def create_ticket(description: str, ticket_type: str = "咨询") -> str:
    """创建工单（写操作）。"""
    return "T123"


calls = {"n": 0}
await_count = {"n": 0}


def reg(*tools):
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


def call(name, args=None, id="c1"):
    return {"name": name, "args": args or {}, "id": id}


async def run(tool_call, registry, **kw):
    return await execute(tool_call, registry, conversation_id=None, **kw)


# ---------- 成功与格式化 ----------

async def test_success_formats_dict_drops_internal_translates_enum():
    out = await run(call("dict_tool", {"order_id": "SO-1"}), reg(dict_tool))
    assert out.status == "成功" and isinstance(out.message, ToolMessage)
    data = json.loads(out.message.content)
    assert data["status"] == "已发货"      # 枚举码翻人话
    assert "recall_score" not in data      # 内部字段剔除
    assert "云优选" in out.message.content  # 中文不转义


def test_format_tool_result_string_passthrough():
    assert format_tool_result("x", "原文") == "原文"


def test_format_unwraps_mcp_content_blocks():
    blocks = [{"type": "text", "text": '{"status": "运输中"}', "id": "lc_x"}]
    out = format_tool_result("query_logistics", blocks)
    assert out == '{"status": "运输中"}'  # 解包成纯文本,不带块结构/内部 id


# ---------- 重试策略 ----------

async def test_transient_network_error_retries_then_succeeds():
    calls["n"] = 0
    out = await run(call("flaky_tool", {"order_id": "1"}), reg(flaky_tool), max_retries=2)
    assert out.status == "成功" and out.retry_count == 2
    assert calls["n"] == 3


@tool
def httpx_flaky_tool(order_id: str) -> str:
    """httpx 网络抖动后成功。"""
    import httpx

    calls["n"] += 1
    if calls["n"] == 1:
        raise httpx.ConnectError("MCP 连接拒绝")
    return "恢复成功"


async def test_httpx_network_error_retries():
    # 评审实测坑:httpx.ConnectError 不继承内置 ConnectionError——白名单漏列则 MCP 故障不重试
    calls["n"] = 0
    out = await run(call("httpx_flaky_tool", {"order_id": "1"}), reg(httpx_flaky_tool), max_retries=1)
    assert out.status == "成功" and out.retry_count == 1


async def test_read_timeout_retries_and_reports():
    await_count["n"] = 0
    out = await run(call("slow_tool", {"order_id": "1"}), reg(slow_tool),
                    default_timeout=0.05, max_retries=2)
    assert out.status == "超时" and out.retry_count == 2
    assert await_count["n"] == 3  # 真重试了
    assert "超时" in out.message.content and "不要编造" in out.message.content


@tool
def slow_write(description: str) -> str:
    """慢写。"""
    import time

    time.sleep(0.2)
    return "T1"


async def test_write_timeout_zero_retries():
    r = ToolRegistry()
    r.register(slow_write, permission="write")  # 显式按写注册(绕开白名单名限制)
    out = await run(call("slow_write", {"description": "x"}), r,
                    default_timeout=0.05, max_retries=2, confirmed=True)
    assert out.status == "超时" and out.retry_count == 0  # 写操作不自动重试(验收⑥)
    assert "不要重复发起" in out.message.content or "勿重复" in out.message.content


async def test_business_error_no_retry():
    out = await run(call("broken_tool", {"order_id": "1"}), reg(broken_tool), max_retries=2)
    assert out.status == "失败" and out.retry_count == 0
    assert "ValueError" in out.message.content  # 真故障如实回


# ---------- 参数校验拦下 ----------

async def test_validation_block_feeds_back_chinese():
    out = await run(call("ok_tool", {}), reg(ok_tool))  # 缺 keyword
    assert out.status == "校验拦下"
    assert "必填" in out.message.content and "keyword" in out.message.content


# ---------- 权限门 ----------

async def test_write_first_visit_returns_pending_not_executed():
    out = await run(call("create_ticket", {"description": "耳机坏了"}), reg(create_ticket))
    assert isinstance(out, PendingWrite)
    assert out.call["name"] == "create_ticket"


async def test_write_cancelled_is_permission_denied():
    out = await run(call("create_ticket", {"description": "x"}), reg(create_ticket), cancelled=True)
    assert out.status == "权限拒绝"
    assert "取消" in out.message.content and "不要再次发起" in out.message.content


async def test_write_confirmed_executes():
    out = await run(call("create_ticket", {"description": "x"}), reg(create_ticket), confirmed=True)
    assert out.status == "成功" and "T123" in out.message.content


async def test_unknown_tool_honest_message():
    out = await run(call("nope"), reg(ok_tool))
    assert out.status == "失败" and "未知工具" in out.message.content


# ---------- 审计(真库) ----------

@pytest.fixture(autouse=True)
def _db(require_db):
    pass


async def test_audit_rows_cover_all_statuses():
    from app.db import get_session_factory
    from app.models import ToolAuditLog
    from sqlalchemy import text

    factory = get_session_factory()
    # 先清本测试会话的旧审计行(隔离)
    with factory() as s:
        s.execute(text("DELETE FROM tool_audit_logs WHERE conversation_id = 424242"))
        s.commit()

    async def go(registry, **kw):
        return await execute(call("ok_tool", {"keyword": "x"}, id="c-audit"), registry,
                             conversation_id=424242, session_factory=factory, **kw)

    await go(reg(ok_tool))                                    # 成功
    await execute(call("ok_tool", {}, id="c-audit2"), reg(ok_tool),
                  conversation_id=424242, session_factory=factory)  # 校验拦下
    await execute(call("create_ticket", {"description": "x"}, id="c-audit3"), reg(create_ticket),
                  conversation_id=424242, session_factory=factory, cancelled=True)  # 权限拒绝
    await execute(call("broken_tool", {"order_id": "1"}, id="c-audit4"), reg(broken_tool),
                  conversation_id=424242, session_factory=factory)  # 失败
    await execute(call("slow_tool", {"order_id": "1"}, id="c-audit5"), reg(slow_tool),
                  conversation_id=424242, session_factory=factory,
                  default_timeout=0.05, max_retries=1)  # 超时(重试1)

    with factory() as s:
        rows = s.execute(text(
            "SELECT tool_call_id, status, retry_count, duration_ms, source FROM tool_audit_logs "
            "WHERE conversation_id=424242 ORDER BY id")).all()
    by_status = {r[1]: r for r in rows}
    assert set(by_status) == {"成功", "校验拦下", "权限拒绝", "失败", "超时"}
    assert by_status["超时"][2] == 1  # retry_count
    assert all(r[3] >= 0 for r in rows)
    assert all(r[4] == "builtin" for r in rows)


async def test_audit_failure_never_blocks_execution():
    class BrokenFactory:
        def __call__(self):
            raise RuntimeError("库挂了")

    out = await execute(call("ok_tool", {"keyword": "x"}), reg(ok_tool),
                        conversation_id=1, session_factory=BrokenFactory())
    assert out.status == "成功"  # 审计写炸了,工具照常返回
