"""电商域工具：query_order / query_product / query_logistics。

ch06 修复轮起统一订单源：订单类工具查 orders 表（按会话定位当前用户，查不到
如实说没有，不再随机造数——消除"物流认识 1001、退款不认识 1001"的两套数据割裂，
用户拍板）。商品查询用静态演示目录（与懒播种商品/知识库商品系列一致）。
物流轨迹按订单状态合成演示数据；随机量用订单号做种子，同一单每次查询结果稳定。
"""

import json

from langchain_core.tools import tool
from sqlalchemy.orm import Session, sessionmaker

from app.agent.orders import get_or_seed_user_orders, match_order_ref
from app.models import Conversation, Order

PRODUCT_CATALOG = [
    {"product_name": "电饭煲 CFXB-40A（4L）", "price": 299.00, "stock": 120, "category": "家用电器"},
    {"product_name": "无线降噪耳机 E-3", "price": 499.00, "stock": 80, "category": "数码3C"},
    {"product_name": "空气净化器 KJ-450F", "price": 1599.00, "stock": 45, "category": "家用电器"},
    {"product_name": "净水器 JR-600", "price": 899.00, "stock": 60, "category": "家用电器"},
    {"product_name": "电饭煲配件蒸笼（通用）", "price": 39.90, "stock": 200, "category": "日用百货"},
]


def _user_orders(session_factory: sessionmaker[Session], conversation_id: int) -> list[Order]:
    """会话 → 当前用户订单（懒播种兜底；会话定位不到用户则空）。"""
    with session_factory() as session:
        conv = session.get(Conversation, conversation_id)
        user_id = conv.user_id if conv else ""
        if not user_id:
            return []
        return get_or_seed_user_orders(session, user_id)


def make_list_orders(session_factory: sessionmaker[Session], conversation_id: int):
    @tool
    def list_my_orders() -> str:
        """列出当前用户名下的全部订单：订单号、商品、金额、状态、下单时间。

        用户问「我有哪些订单 / 我的订单 / 帮我看看订单」这类不带具体单号的诉求时调用；
        按单号查详情/物流请改用 query_order / query_logistics。
        """
        orders = _user_orders(session_factory, conversation_id)
        if not orders:
            return json.dumps({"orders": [], "note": "当前用户名下没有订单"}, ensure_ascii=False)
        return json.dumps(
            {
                "orders": [
                    {
                        "order_id": o.order_no,
                        "product_name": o.product_name,
                        "amount": float(o.amount),
                        "status": o.status,
                        "created_at": o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "",
                    }
                    for o in orders
                ]
            },
            ensure_ascii=False,
        )

    return list_my_orders


def make_query_order(session_factory: sessionmaker[Session], conversation_id: int):
    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询当前用户的订单信息：状态、金额、商品、下单时间。查不到会如实返回未找到。

        Args:
            order_id: 订单号（完整单号如 SO-XXXXXX-02，或末尾短号如 02）
        """
        orders = _user_orders(session_factory, conversation_id)
        order = match_order_ref(orders, order_id)
        if order is None:
            return json.dumps(
                {
                    "error": f"当前用户名下没有订单 {order_id}",
                    "available_orders": [o.order_no for o in orders],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "order_id": order.order_no,
                "status": order.status,
                "amount": float(order.amount),
                "product_name": order.product_name,
                "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else "",
            },
            ensure_ascii=False,
        )

    return query_order


@tool
def query_product(keyword: str) -> str:
    """按关键词查询在售商品：名称、价格、库存、分类（静态演示目录）。

    Args:
        keyword: 商品名称关键词，如 电饭煲
    """
    hits = [dict(p) for p in PRODUCT_CATALOG if keyword in p["product_name"]]
    if not hits:
        return json.dumps({"error": f"没有找到与「{keyword}」相关的在售商品"}, ensure_ascii=False)
    return json.dumps({"keyword": keyword, "results": hits}, ensure_ascii=False)
