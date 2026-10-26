"""工具基础设施：注册中心（ch08）。

注册中心：内置/插件/MCP 工具一律登记 ToolRecord（三要素：名称/用途描述/JSON Schema
参数定义 + 来源 + 权限）。权限只认 policy.py 的我方规则表。
执行统一入口是 engine.py（ch08）：权限门/校验/超时重试/分诊/审计全收口,
旧 execute_tool_call 已删除（chat_service/bare_loop/react 均已切引擎）。
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.tools.policy import classify_permission

import logging
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.tools.policy import classify_permission

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    """注册条目：不管内置还是 MCP，三要素带齐才准登记。"""

    name: str
    description: str
    args_schema: dict
    source: str  # builtin | mcp:<server>
    permission: str  # read | write（只认我方 policy，不看 Server 声明）
    tool: BaseTool
    timeout_seconds: float | None = None


def _derive_schema(tool: BaseTool) -> dict:
    """从工具推导完整 JSON Schema（实测坑：本版 tool.args 只给 properties 裸字典，
    缺 type/required，直接当 Schema 用校验等于没开）。

    优先 pydantic args_schema.model_json_schema()（含 required）；
    回退用 tool.args 包装并按 default 推 required；无参工具给空对象 Schema。
    """
    try:
        pyd = getattr(tool, "args_schema", None)
        if pyd is not None and hasattr(pyd, "model_json_schema"):
            schema = pyd.model_json_schema()
            if isinstance(schema, dict) and schema.get("properties") is not None:
                return schema
    except Exception:
        pass
    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        required = [k for k, v in args.items() if not (isinstance(v, dict) and "default" in v)]
        return {"type": "object", "properties": args, "required": required}
    return {"type": "object", "properties": {}}


class ToolRegistry:
    """name -> ToolRecord 注册表；重名策略：内置炸出来（注册事故），MCP 跳过+告警（别留重名工具）。"""

    def __init__(self) -> None:
        self._records: dict[str, ToolRecord] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        source: str = "builtin",
        permission: str | None = None,
        timeout_seconds: float | None = None,
        on_duplicate: str = "raise",
    ) -> ToolRecord | None:
        if tool.name in self._records:
            if on_duplicate == "skip":
                logger.warning(
                    "工具重名跳过: %s（已注册=%s 新来源=%s 不覆盖）",
                    tool.name, self._records[tool.name].source, source,
                )
                return None
            raise ValueError(f"工具重名: {tool.name}")
        record = ToolRecord(
            name=tool.name,
            description=str(tool.description or ""),
            args_schema=_derive_schema(tool),
            source=source,
            permission=permission or classify_permission(tool.name),
            tool=tool,
            timeout_seconds=timeout_seconds,
        )
        self._records[tool.name] = record
        return record

    def get(self, name: str) -> ToolRecord | None:
        return self._records.get(name)

    def records(self) -> list[ToolRecord]:
        return list(self._records.values())

    def all(self) -> list[BaseTool]:
        """bind_tools 视角的工具清单。"""
        return [r.tool for r in self._records.values()]

    def timeout_for(self, name: str, default: float | None = None) -> float | None:
        rec = self._records.get(name)
        if rec is None:
            return default
        return rec.timeout_seconds if rec.timeout_seconds is not None else default
