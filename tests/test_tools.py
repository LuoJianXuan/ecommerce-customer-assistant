"""工具基础设施接缝：注册表（ch08 起执行统一入口在 engine.py,旧 execute_tool_call 已删,
超时/重试/错误降级语义由 tests/test_tool_engine.py 覆盖）。"""

import json

import pytest
from langchain_core.tools import tool as tool_decorator

from app.tools.base import ToolRegistry


def make_echo_tool():
    @tool_decorator
    def echo(text: str) -> str:
        """原样返回文本。"""
        return f"echo:{text}"

    return echo


def test_registry_register_and_get():
    reg = ToolRegistry()
    echo = make_echo_tool()
    rec = reg.register(echo)
    assert rec.name == "echo" and rec.permission == "read"
    assert reg.get("echo").tool is echo
    assert [t.name for t in reg.all()] == ["echo"]


def test_registry_duplicate_name_rejected():
    reg = ToolRegistry()
    reg.register(make_echo_tool())
    with pytest.raises(ValueError):
        reg.register(make_echo_tool())


def test_registry_unknown_tool_returns_none():
    reg = ToolRegistry()
    assert reg.get("nope") is None


def test_ticket_no_format():
    import re

    from app.tools.tickets import gen_ticket_no

    no = gen_ticket_no()
    assert re.fullmatch(r"T\d{8}\d{3}", no), no
