"""ch08 参数校验：工具执行前统一按 JSON Schema 校验。

拦下不抛异常——产出中文错误说明，由调用方包成 ToolMessage 回灌给模型，
让它修正参数重试或向用户澄清（需求原文）。
"""

import jsonschema


def validate_tool_args(args_schema: dict, args) -> str | None:
    """返回 None=通过；返回中文说明=拦下（含具体原因）。"""
    if not isinstance(args, dict):
        return (
            f"参数校验未通过：参数必须是键值对象，本次收到的是 {type(args).__name__}。"
            "请按工具的参数定义重新组织调用。"
        )
    try:
        jsonschema.validate(args, args_schema)
        return None
    except jsonschema.ValidationError as exc:
        field = ".".join(str(p) for p in exc.absolute_path) or "参数"
        if exc.validator == "required":
            return (
                f"参数校验未通过：{exc.message}。"
                "必填字段缺失——请补齐后重试；信息在对话里找不到时，先向用户追问，不要编造。"
            )
        return (
            f"参数校验未通过：{field} {exc.message}。"
            "请修正参数后重试，或向用户澄清后再调用。"
        )
