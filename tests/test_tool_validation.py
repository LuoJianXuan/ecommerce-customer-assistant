"""ch08 参数校验（TDD）：按 JSON Schema 拦下类型错误/必填缺失/取值越界,产出中文回灌说明。

纯函数测试。
"""

from langchain_core.tools import tool

from app.tools.base import ToolRegistry
from app.tools.validation import validate_tool_args


@tool
def lookup(kind: str, limit: int = 5) -> str:
    """查点什么。kind 只能是 订单/物流;limit 1-10。"""
    return f"{kind}:{limit}"


def _schema_of(t):
    reg = ToolRegistry()
    return reg.register(t).args_schema


def test_passes_valid_args():
    assert validate_tool_args(_schema_of(lookup), {"kind": "订单", "limit": 3}) is None


def test_missing_required_field_reports_chinese_reason():
    msg = validate_tool_args(_schema_of(lookup), {"limit": 3})
    assert msg is not None and "kind" in msg and "必填" in msg


def test_wrong_type_reported():
    msg = validate_tool_args(_schema_of(lookup), {"kind": "订单", "limit": "三"})
    assert msg is not None and "limit" in msg


def test_schema_constrains_enum_and_range():
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["订单", "物流"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["kind"],
    }
    assert validate_tool_args(schema, {"kind": "商品"}) is not None  # 枚举越界
    assert validate_tool_args(schema, {"kind": "订单", "limit": 99}) is not None  # 数值越界
    assert validate_tool_args(schema, {"kind": "订单", "limit": 10}) is None


def test_non_object_args_rejected():
    schema = _schema_of(lookup)
    assert validate_tool_args(schema, "不是个对象") is not None
