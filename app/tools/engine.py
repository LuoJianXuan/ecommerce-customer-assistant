"""ch08 执行引擎：所有工具调用的唯一入口。

职责（需求原文）：统一管超时、重试、错误处理、结果格式化 + 审计落库。
- 重试只给暂时性故障（网络类异常/超时）且**只读工具**；业务空结果不重试；
  写操作一律不自动重试（超时未必没执行，重复执行比失败更糟）；
- 错误三诊：参数不合法（校验拦下）/ 查询落空（工具如实返回 not-found，正常回灌不重试）/
  真故障（异常/超时——错误消息如实回给模型，不粉饰）；
- 结果格式化：枚举码翻人话 + 内部字段剔除 + JSON 中文不转义；
- 写操作确认门：WRITE_TOOLS 不直接执行——confirmed=True 才真执行；未确认/取消一律
  权限拒绝 + 审计（不给模型绕过去的机会）；
- 审计：每次调用（含被拦/被拒）落 tool_audit_logs；写审计失败只告警，绝不反拦执行。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
from langchain_core.messages import ToolMessage

from app.tools.base import ToolRegistry
from app.tools.policy import classify_permission
from app.tools.validation import validate_tool_args

logger = logging.getLogger("tool_engine")

# 暂时性故障白名单：仅这些（+超时）对只读工具重试。
# httpx 异常族必须显式列入——httpx.ConnectError 不继承内置 ConnectionError（评审实测 MRO），
# 漏了的话 MCP 工具的网络故障实际不会重试
TRANSIENT_ERRORS = (ConnectionError, httpx.TransportError)

# 内部枚举码 → 人话（机制在，现网数据多为中文；命中才翻）
ENUM_TRANSLATIONS = {
    "shipped": "已发货", "pending": "待付款", "paid": "已付款",
    "completed": "已签收", "cancelled": "已取消", "in_warranty": "在保", "expired": "已过保",
}

# 剔除的内部字段（对回答无用且占上下文）
INTERNAL_RESULT_KEYS = {"recall_score", "vector_id", "section_path"}

# 字段白名单（自裁决点②）：默认空=全保留；按工具名配置只保留白名单字段
TOOL_FIELD_WHITELIST: dict[str, set[str]] = {}


@dataclass
class ToolExecution:
    message: ToolMessage      # 回灌给模型的消息
    status: str               # 审计状态：成功/失败/超时/校验拦下/权限拒绝
    error: str | None = None
    retry_count: int = 0
    duration_ms: int = 0
    result_summary: str = ""  # 审计用结果摘要（前 500 字）


@dataclass
class PendingWrite:
    """写操作挂起：引擎不执行，交调用方走确认流（M4 的 ticket_confirm 节点）。"""

    call: dict                # 原始 tool_call {name, args, id}
    preview: dict             # 前端预览数据


def _translate(value):
    """枚举码翻人话（递归 dict/list/str）。"""
    if isinstance(value, str):
        return ENUM_TRANSLATIONS.get(value, value)
    if isinstance(value, list):
        return [_translate(v) for v in value]
    if isinstance(value, dict):
        return {k: _translate(v) for k, v in value.items()}
    return value


def _prune(name: str, result):
    if not isinstance(result, dict):
        return result
    whitelist = TOOL_FIELD_WHITELIST.get(name)
    if whitelist:
        return {k: v for k, v in result.items() if k in whitelist}
    return {k: v for k, v in result.items() if k not in INTERNAL_RESULT_KEYS}


def _unwrap_content_blocks(result):
    """MCP 工具返回的是 content-block 列表（[{'type':'text','text':...}]）——解包拼纯文本，
    不让模型看到块结构与内部 id。"""
    if (
        isinstance(result, list)
        and result
        and all(isinstance(i, dict) and i.get("type") == "text" for i in result)
    ):
        return "\n".join(str(i.get("text", "")) for i in result)
    return result


def format_tool_result(name: str, result) -> str:
    """结果格式化：MCP 块解包 → 裁剪 → 枚举翻人话 → JSON 中文不转义。"""
    result = _unwrap_content_blocks(result)
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    cleaned = _translate(_prune(name, result))
    return json.dumps(cleaned, ensure_ascii=False)


def write_audit(session_factory, *, conversation_id, tool_call_id, tool_name, source,
                params, result_summary, status, error, retry_count, duration_ms) -> None:
    """审计落库：不挂外键；写失败只告警，绝不反拦工具执行。"""
    if session_factory is None:
        return
    try:
        from app.models import ToolAuditLog

        with session_factory() as session:
            session.add(ToolAuditLog(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                source=source,
                params=params if isinstance(params, dict) else None,
                result_summary=(result_summary or "")[:500] or None,
                status=status,
                error=(error or "")[:500] or None,
                retry_count=retry_count,
                duration_ms=duration_ms,
            ))
            session.commit()
    except Exception:
        logger.warning("审计写入失败（不影响工具执行）: %s %s", tool_name, status, exc_info=True)


def _error_msg(tool_call_id, name, text) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=tool_call_id, name=name, status="error")


async def execute(
    tool_call: dict,
    registry: ToolRegistry,
    *,
    conversation_id: int | None,
    session_factory=None,
    default_timeout: float = 5.0,
    max_retries: int = 1,
    confirmed: bool = False,
    cancelled: bool = False,
    allow_pending: bool = True,
) -> ToolExecution | PendingWrite:
    """统一执行入口。

    返回 ToolExecution（含回灌消息与审计信息）或 PendingWrite（写操作首访，等确认流）；
    confirmed=True 的写操作真执行；cancelled=True 或未确认直调 → 权限拒绝。
    allow_pending=False 用于没有确认 UI 的旧通道（ch02 /api/chat、bare_loop）：写操作
    一律权限拒绝+审计，绝无绕过确认落库的口子。审计在函数内完成（含被拦/被拒），失败不外抛。
    """
    name = tool_call.get("name", "")
    tool_call_id = tool_call.get("id", "")
    args = tool_call.get("args") or {}
    started = time.perf_counter()
    record = registry.get(name)

    def _audit(status, error=None, summary="", retries=0):
        write_audit(session_factory, conversation_id=conversation_id, tool_call_id=tool_call_id,
                    tool_name=name or "未知", source=record.source if record else "unknown",
                    params=args, result_summary=summary, status=status, error=error,
                    retry_count=retries, duration_ms=int((time.perf_counter() - started) * 1000))

    if record is None:
        msg = f"未知工具: {name}。请从可用工具清单中选择，或如实告知用户当前无法处理该请求。"
        _audit("失败", error="未知工具")
        return ToolExecution(_error_msg(tool_call_id, name, msg), "失败", error="未知工具")

    # ---- 权限门：写操作必须经确认流 ----
    if record.permission == "write":
        if cancelled:
            msg = "权限拒绝：用户已取消本次操作，未执行。请向用户确认后续意向，不要再次发起同一调用。"
            _audit("权限拒绝", error="用户取消")
            return ToolExecution(_error_msg(tool_call_id, name, msg), "权限拒绝", error="用户取消")
        if not confirmed:
            # 写操作业务前置：缺关键信息(如空描述)直接拦下,逼模型先追问(需求:不许瞎编)
            from app.tools.policy import preflight_write

            violation = preflight_write(name, args)
            if violation is not None:
                _audit("校验拦下", error=violation[:200])
                return ToolExecution(_error_msg(tool_call_id, name, violation), "校验拦下", error=violation)
            # 首次到达：不执行、不审计（尚无调用事实），交确认流；取消回来才落权限拒绝
            if not allow_pending:
                msg = (
                    "权限拒绝：写操作必须经前端确认后执行，当前调用通道不支持确认流，已拒绝。"
                    "请告知用户在对话页发起该操作。"
                )
                _audit("权限拒绝", error="通道不支持确认流")
                return ToolExecution(_error_msg(tool_call_id, name, msg), "权限拒绝", error="通道不支持确认流")
            return PendingWrite(call=dict(tool_call), preview={})
        # confirmed=True → 放行执行（写操作下方不自动重试）

    # ---- 参数校验：拦下回灌，不抛异常 ----
    invalid = validate_tool_args(record.args_schema, args)
    if invalid is not None:
        _audit("校验拦下", error=invalid[:200])
        return ToolExecution(_error_msg(tool_call_id, name, invalid), "校验拦下", error=invalid)

    # ---- 执行（超时/重试/分诊） ----
    timeout = registry.timeout_for(name, default_timeout)
    is_read = record.permission == "read"
    attempts = (max_retries + 1) if is_read else 1  # 写操作默认不自动重试
    retries_done = 0
    last_kind, last_err = "失败", ""
    result = None
    for attempt in range(attempts):
        try:
            raw = await asyncio.wait_for(record.tool.ainvoke(args), timeout=timeout)
            content = format_tool_result(name, raw)
            _audit("成功", summary=content)
            return ToolExecution(
                ToolMessage(content=content, tool_call_id=tool_call_id, name=name, status="success"),
                "成功", result_summary=content,
                retry_count=retries_done,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except asyncio.TimeoutError:
            last_kind, last_err = "超时", f"执行超时（>{timeout}s）"
        except TRANSIENT_ERRORS as exc:
            last_kind, last_err = "失败", f"网络类暂时性故障: {type(exc).__name__}: {exc}"
        except Exception as exc:  # 业务异常/参数在 invoke 内被拒 → 真故障或业务错,不重试
            last_kind, last_err = "失败", f"{type(exc).__name__}: {exc}"
            break
        if attempt + 1 < attempts:
            retries_done = attempt + 1  # 只有真会再试才算一次重试(最后一次失败不计)
    msg = f"工具 {name} {last_kind}: {last_err}。" + (
        "请稍后重试或换一种方式回答，不要编造结果。" if is_read else "写操作不确定是否已执行，请勿重复发起，建议向用户说明并转人工核实。"
    )
    _audit(last_kind, error=last_err, retries=retries_done)
    return ToolExecution(
        _error_msg(tool_call_id, name, msg), last_kind, error=last_err,
        retry_count=retries_done, duration_ms=int((time.perf_counter() - started) * 1000),
    )
