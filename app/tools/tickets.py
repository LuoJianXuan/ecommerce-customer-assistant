"""工单工具：create_ticket，写 tickets 表（工单号 T+日期+3位序号）。

conversation_id 由聊天服务在构建注册表时经闭包注入（会话行在轮次开始即创建），
满足 fk_tickets_conversation 外键。
"""

import random
from datetime import date

from sqlalchemy.orm import Session, sessionmaker
from langchain_core.tools import tool

from app.models import Ticket


def gen_ticket_no() -> str:
    """生成工单号：T + YYYYMMDD + 3 位随机序号，如 T202609280137。"""
    return f"T{date.today():%Y%m%d}{random.randint(0, 999):03d}"


def make_create_ticket(session_factory: sessionmaker[Session], conversation_id: int):
    @tool
    def create_ticket(
        description: str,
        ticket_type: str = "咨询",
        order_id: str = "",
    ) -> str:
        """创建人工处理工单（写操作，需用户在前端确认后才真正执行）。仅在用户明确要求建工单/转人工跟进时调用；
        问题描述缺失时不要调用，先向用户追问清楚（问题是什么、涉及什么订单/商品），不许编造。
        用户只是抱怨但没要求建单时不要调用。

        Args:
            description: 问题描述，概括用户诉求与背景（必填，来自用户原话）
            ticket_type: 工单类型，只能是 售后/投诉/咨询 之一
            order_id: 相关订单号，用户提到才填，可为空
        """
        ticket_no = gen_ticket_no()
        full_description = description + (f"（订单号: {order_id}）" if order_id else "")
        with session_factory() as session:
            session.add(
                Ticket(
                    ticket_no=ticket_no,
                    conversation_id=conversation_id,
                    description=full_description,
                    ticket_type=ticket_type,
                )
            )
            session.commit()
        return f"工单已创建，工单号 {ticket_no}，人工客服会尽快跟进处理。"

    return create_ticket
