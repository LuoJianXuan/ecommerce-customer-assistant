"""ch06 订单域（TDD）：懒播种幂等、订单归属、退款单创建与校验、退款 API。

需要 MySQL（require_db），不碰 LLM 与 Milvus。
"""

import uuid

import pytest

from app.agent.orders import (
    ORDER_STATUSES,
    REFUND_REASONS,
    create_refund,
    extract_order_refs,
    gen_order_no,
    get_or_seed_user_orders,
    match_order_ref,
)
from app.db import get_session_factory
from app.models import Conversation, RefundRequest


@pytest.fixture(autouse=True)
def _db(require_db):
    pass


@pytest.fixture
def session():
    s = get_session_factory()()
    yield s
    s.close()


def _new_user() -> str:
    return "u-test-" + uuid.uuid4().hex[:12]


# ---------- 懒播种 ----------

def test_seed_creates_demo_orders_for_fresh_user(session):
    user = _new_user()
    orders = get_or_seed_user_orders(session, user)
    assert 3 <= len(orders) <= 5
    assert all(o.user_id == user for o in orders)
    assert all(o.status in ORDER_STATUSES for o in orders)
    assert all(o.order_no.startswith("SO-") for o in orders)


def test_seed_idempotent_same_orders_on_second_call(session):
    user = _new_user()
    first = [o.order_no for o in get_or_seed_user_orders(session, user)]
    second = [o.order_no for o in get_or_seed_user_orders(session, user)]
    assert sorted(first) == sorted(second)  # 已有订单不再播种


def test_gen_order_no_deterministic_per_user():
    assert gen_order_no("u-abc", 1) == gen_order_no("u-abc", 1)
    assert gen_order_no("u-abc", 1) != gen_order_no("u-abc", 2)
    assert gen_order_no("u-abc", 1) != gen_order_no("u-xyz", 1)


# ---------- 退款单 ----------

def test_create_refund_happy_path(session):
    user = _new_user()
    order = get_or_seed_user_orders(session, user)[0]
    refund_no = create_refund(session, order.id, user, "七天无理由")
    assert refund_no.startswith("RF-")
    row = session.query(RefundRequest).filter_by(refund_no=refund_no).one()
    assert row.order_id == order.id
    assert row.status == "待审核"


def test_create_refund_rejects_invalid_reason(session):
    user = _new_user()
    order = get_or_seed_user_orders(session, user)[0]
    with pytest.raises(ValueError):
        create_refund(session, order.id, user, "随便写的理由")


def test_create_refund_rejects_foreign_order(session):
    owner, attacker = _new_user(), _new_user()
    order = get_or_seed_user_orders(session, owner)[0]
    with pytest.raises(ValueError):
        create_refund(session, order.id, attacker, "七天无理由")  # 不能退别人的单


def test_refund_reasons_are_fixed_categories():
    assert REFUND_REASONS == ("七天无理由", "质量问题", "与描述不符", "少件漏发", "不想要了", "其他")


# ---------- 订单号槽位提取（修复轮） ----------

def test_extract_order_refs_full_and_short_numbers():
    assert extract_order_refs("SO-AB12CD-02 能退吗") == ["SO-AB12CD-02"]
    assert extract_order_refs("我订单1001到哪了") == ["1001"]
    got = extract_order_refs("so-ab12cd-02 和 SO-AB12CD-03 哪个能退")
    assert got == ["so-ab12cd-02", "SO-AB12CD-03"]


def test_extract_order_refs_ignores_short_numbers_dedupes():
    assert extract_order_refs("7天无理由还能退吗") == []  # 1-2 位数字不误伤
    assert extract_order_refs("订单 1001 还是 1001 那单") == ["1001"]


def test_match_order_ref_exact_case_insensitive_and_suffix(session):
    user = _new_user()
    orders = get_or_seed_user_orders(session, user)
    no = orders[0].order_no  # SO-XXXXXX-NN
    assert match_order_ref(orders, no).order_no == no
    assert match_order_ref(orders, no.lower()) is not None
    suffix = no.rsplit("-", 1)[-1]
    assert match_order_ref(orders, suffix).order_no == no
    assert match_order_ref(orders, "1001") is None  # 不存在的短号:如实未命中
    assert match_order_ref(orders, "") is None


# ---------- 退款 API ----------

def test_refund_api_roundtrip(client, session):
    user = _new_user()
    order = get_or_seed_user_orders(session, user)[0]
    conv = Conversation(user_id=user)
    session.add(conv)
    session.commit()

    resp = client.post(
        "/api/agent/refund",
        json={"conversation_id": conv.id, "user_id": user, "order_id": order.id, "reason": "质量问题"},
    )
    assert resp.status_code == 200
    assert resp.json()["refund_no"].startswith("RF-")
    assert resp.json()["status"] == "待审核"

    bad = client.post(
        "/api/agent/refund",
        json={"conversation_id": conv.id, "user_id": user, "order_id": order.id, "reason": "不存在的理由"},
    )
    assert bad.status_code == 422

    missing = client.post(
        "/api/agent/refund",
        json={"conversation_id": 999999999, "user_id": user, "order_id": order.id, "reason": "质量问题"},
    )
    assert missing.status_code == 404
