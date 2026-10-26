"""HTTP 接缝：POST /api/after-sale/extract 结构化提取。"""

import pytest

from app.schemas import AfterSaleInfo
from tests.conftest import make_client
from tests.fakes import FakeStructuredLLM

AFTER_SALE_TEXT = "我3天前买的订单SO202609190001，收到的电饭煲内胆有裂纹，要求给我换一个全新的。"


@pytest.fixture
def extract_setup():
    fake = FakeStructuredLLM(
        AfterSaleInfo(order_id="SO202609190001", appeal_type="换货", expected_solution="换一个全新的")
    )
    return make_client(fake), fake


def test_extract_returns_fixed_fields(extract_setup):
    client, fake = extract_setup
    resp = client.post("/api/after-sale/extract", json={"text": AFTER_SALE_TEXT})

    assert resp.status_code == 200
    assert resp.json() == {
        "order_id": "SO202609190001",
        "appeal_type": "换货",
        "expected_solution": "换一个全新的",
    }
    # 用户原文要进到提取 prompt 里
    assert "电饭煲" in str(fake.calls[0])


def test_extract_keeps_null_fields(extract_setup):
    client, fake = extract_setup
    fake.result = AfterSaleInfo()  # 换成全空结果
    resp = client.post("/api/after-sale/extract", json={"text": "我就是吐槽一下"})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"order_id", "appeal_type", "expected_solution"}
    assert all(v is None for v in body.values())


def test_extract_rejects_empty_text(extract_setup):
    client, _ = extract_setup
    resp = client.post("/api/after-sale/extract", json={"text": ""})
    assert resp.status_code == 422
