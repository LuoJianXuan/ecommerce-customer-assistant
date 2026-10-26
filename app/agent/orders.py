"""ch06 订单域服务：懒播种演示订单 + 退款单创建。

订单选择器/退款子流程的唯一数据源（订单不再走工具随机造数）。
懒播种（用户拍板）：该 user_id 无订单时幂等生成 3-5 条演示订单——
  - order_no 由 user_id 哈希派生（确定性），并发双播种撞 UNIQUE 约束回滚即可；
  - 商品与状态取自知识库商品系列，时间在近 30 天内。
退款单：原因固定类目（表单下拉），后端再校验；不能退别人的单。
"""

import hashlib
import random
import re
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Order, RefundRequest

ORDER_STATUSES = ("待付款", "已付款", "已发货", "已签收", "已取消")
REFUND_REASONS = ("七天无理由", "质量问题", "与描述不符", "少件漏发", "不想要了", "其他")

# 订单号引用：SO- 全格式（忽略大小写），或 3-8 位纯数字短号（"7天"这类 1-2 位数字不误伤）
ORDER_REF_RE = re.compile(r"SO-[0-9A-Za-z]{2,10}(?:-[0-9]{1,4})?|\d{3,8}", re.IGNORECASE)

# 演示订单模板：商品对齐知识库商品系列，状态覆盖典型退款场景
_DEMO_TEMPLATES = [
    ("电饭煲 CFXB-40A（4L）", "已签收", 299.00),
    ("无线降噪耳机 E-3", "已发货", 499.00),
    ("空气净化器 KJ-450F", "已付款", 1599.00),
    ("净水器 JR-600", "待付款", 899.00),
]


def gen_order_no(user_id: str, seq: int) -> str:
    """演示订单号：SO-<user哈希6位>-<序号>。确定性——同一用户重播结果相同。"""
    tag = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:6].upper()
    return f"SO-{tag}-{seq:02d}"


def gen_refund_no() -> str:
    """退款单号：RF-YYYYMMDD-NNN（3 位随机序号）。"""
    return f"RF-{date.today():%Y%m%d}-{random.randint(0, 999):03d}"


def list_user_orders(session: Session, user_id: str) -> list[Order]:
    stmt = select(Order).where(Order.user_id == user_id).order_by(Order.created_at.desc())
    return list(session.scalars(stmt))


def get_or_seed_user_orders(session: Session, user_id: str) -> list[Order]:
    """查该用户订单；没有则懒播种演示数据（幂等，并发安全靠 order_no 唯一约束）。"""
    orders = list_user_orders(session, user_id)
    if orders:
        return orders
    rng = random.Random(user_id)  # 同一用户稳定复现同一批订单
    now = datetime.now()
    for seq, (product, status, amount) in enumerate(_DEMO_TEMPLATES, 1):
        created = now - timedelta(days=rng.randint(1, 30))
        session.add(
            Order(
                order_no=gen_order_no(user_id, seq),
                user_id=user_id,
                product_name=product,
                amount=amount,
                status=status,
                created_at=created,
                updated_at=created,
            )
        )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()  # 并发双播种：约束兜底，读现存即可
    return list_user_orders(session, user_id)


def create_refund(session: Session, order_id: int, user_id: str, reason: str) -> str:
    """创建退款单，返回 refund_no。原因必须属于固定类目；订单必须存在且属于该用户。"""
    if reason not in REFUND_REASONS:
        raise ValueError(f"退款原因必须是固定类目之一: {REFUND_REASONS}")
    order = session.get(Order, order_id)
    if order is None or order.user_id != user_id:
        raise ValueError("订单不存在或不属于当前用户")
    refund_no = gen_refund_no()
    session.add(
        RefundRequest(refund_no=refund_no, order_id=order_id, user_id=user_id, reason=reason)
    )
    session.commit()
    return refund_no


def order_summary(order: Order) -> dict:
    """订单 → 订单选择器卡片字段。"""
    return {
        "id": order.id,
        "order_no": order.order_no,
        "product_name": order.product_name,
        "amount": float(order.amount),
        "status": order.status,
        "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else "",
    }


# ---------- 订单号槽位提取（ch06 修复轮：子流程/工具都要消费用户点名的订单号） ----------

def extract_order_refs(text: str) -> list[str]:
    """从文本提取订单号引用（确定性正则，不让模型猜）：SO- 全格式或 3-8 位纯数字。

    去重保序，最多 3 个；提取本身不做存在性判断（交给 match_order_ref）。
    """
    refs, seen = [], set()
    for m in ORDER_REF_RE.finditer(str(text or "")):
        r = m.group(0)
        if r not in seen:
            refs.append(r)
            seen.add(r)
        if len(refs) >= 3:
            break
    return refs


def match_order_ref(orders: list[Order], ref: str) -> Order | None:
    """订单号引用 → 用户订单：先全号精确匹配（忽略大小写），再短号后缀段匹配。

    例："SO-AB12CD-02" 精确命中；"02" 命中 *-02；"1001" 命不中任何 SO- 单（如实未找到）。
    """
    r = str(ref).strip().upper()
    if not r:
        return None
    for o in orders:
        if o.order_no.upper() == r:
            return o
    for o in orders:
        if o.order_no.upper().endswith("-" + r):
            return o
    return None
