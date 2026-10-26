"""ch08 工具注册中心（TDD）：ToolRecord 三要素、权限默认、重名策略、插件自动扫描。

纯内存测试，不碰 DB/LLM。
"""

import pytest
from langchain_core.tools import tool

from app.tools.base import ToolRegistry
from app.tools.policy import WRITE_TOOLS, classify_permission


@tool
def echo(text: str) -> str:
    """原样返回文本。"""
    return f"echo:{text}"


@tool
def create_ticket(description: str, ticket_type: str = "咨询") -> str:
    """创建工单。"""
    return "ok"


@tool
def bounded(kind: str, limit: int = 5) -> str:
    """带枚举与数值参数的工具。"""
    return f"{kind}:{limit}"


def test_record_carries_three_elements_and_metadata():
    reg = ToolRegistry()
    rec = reg.register(echo)
    assert rec.name == "echo"
    assert rec.description == "原样返回文本。"
    assert rec.args_schema["type"] == "object"
    assert "text" in rec.args_schema["properties"]  # JSON Schema 参数定义
    assert rec.source == "builtin"
    assert reg.all() == [echo]  # bind_tools 视角仍是 BaseTool 清单


def test_permission_comes_from_our_rules_not_declaration():
    reg = ToolRegistry()
    assert reg.register(echo).permission == "read"
    assert reg.register(create_ticket).permission == "write"  # 只认我方白名单
    assert classify_permission("query_order") == "read"
    assert WRITE_TOOLS == {"create_ticket"}


def test_duplicate_builtin_raises_mcp_skip_warns(caplog):
    reg = ToolRegistry()
    reg.register(echo)
    with pytest.raises(ValueError):
        reg.register(echo)  # 内置重名=注册事故,直接炸
    with caplog.at_level("WARNING"):
        rec = reg.register(echo, source="mcp:demo", on_duplicate="skip")
    assert rec is None  # MCP 重名跳过,不覆盖内置
    assert any("重名" in r.message for r in caplog.records)


def test_timeout_override_per_record():
    reg = ToolRegistry()
    reg.register(echo, timeout_seconds=12.0)
    assert reg.timeout_for("echo") == 12.0
    reg.register(create_ticket)
    assert reg.timeout_for("create_ticket") is None  # 无覆盖→调用方默认


# ---------- 插件自动扫描(拍板①) ----------

def test_plugin_scan_discovers_demo_plugin():
    from app.tools.plugins import discover_plugins

    tools, factories = discover_plugins()
    names = [t.name for t in tools]
    assert "query_coupon" in names  # 示例插件被启动扫描发现
    for f in factories:
        assert callable(f)  # FACTORIES 条目都是可调用工厂


def test_demo_plugin_registered_via_build_default_registry():
    from app.tools import build_default_registry

    reg = build_default_registry(None, 0)
    rec = reg.get("query_coupon")
    assert rec is not None and rec.source == "builtin" and rec.permission == "read"
    out = rec.tool.invoke({"keyword": "耳机"})
    assert "优惠券" in out or "coupon" in out.lower()
