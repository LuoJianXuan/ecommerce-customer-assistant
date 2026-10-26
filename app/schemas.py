"""API 请求/响应模型与结构化输出 schema。"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(default="web-anon", min_length=1, max_length=64, description="用户标识，落 conversations.user_id")
    conversation_id: int | None = Field(default=None, description="会话 id；缺省则新建会话，流首帧 meta 返回真实 id")
    message: str = Field(min_length=1, max_length=4000, description="本轮用户输入")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000, description="用户的售后问题描述原文")


class AfterSaleInfo(BaseModel):
    """售后诉求结构化字段（with_structured_output 的目标 schema）。"""

    order_id: str | None = Field(
        default=None, description="订单号。用户原文中提到才提取，格式如 SO 开头的编号；没提到则为 null"
    )
    appeal_type: Literal[
        "退款", "退货退款", "换货", "补发", "维修", "降价补偿", "投诉", "其他"
    ] | None = Field(
        default=None, description="诉求类型。用户没表达明确售后诉求时为 null"
    )
    expected_solution: str | None = Field(
        default=None,
        description="期望方案。用户期望的具体处理方式，尽量用用户原话概括；没提到则为 null",
    )
