"""HTTP 接缝：/api/kb/* 知识库管理接口（真库集成）。"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def kb_client(require_db):
    return TestClient(create_app())


def test_kb_page_served(kb_client):
    resp = kb_client.get("/kb")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "知识库" in resp.text


def test_kb_stats_shape(kb_client):
    resp = kb_client.get("/api/kb/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "total", "done", "pending",
        "staging_kept", "staging_discarded", "staging_extracted",
        "milvus_ok",
    }
    assert body["total"] >= body["pending"]


def test_kb_chunks_list_and_filters(kb_client):
    resp = kb_client.get("/api/kb/chunks")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items and {"id", "category", "questions", "answer", "content_type",
                      "is_key_clause", "vectorize_status", "section_path"} <= set(items[0])

    # 搜索过滤：用已知存在的词
    hit = kb_client.get("/api/kb/chunks", params={"q": "运费"}).json()["items"]
    assert hit and all("运费" in (i["questions"] + i["answer"]) for i in hit)
    # 分类过滤
    cats = kb_client.get("/api/kb/chunks", params={"category": "退货政策"}).json()["items"]
    assert cats and all(i["category"] == "退货政策" for i in cats)


def test_kb_preview_no_persist(kb_client):
    md = "# 测试手册\n## 首节\n第一段内容。\n## 次节\n第二段内容。"
    resp = kb_client.post("/api/kb/preview", json={"markdown": md})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert {c["questions"] for c in body["chunks"]} == {"首节", "次节"}
    # 预览不落库
    gone = kb_client.get("/api/kb/chunks", params={"q": "第二段内容"}).json()["items"]
    assert all(i["category"] != "测试手册" for i in gone)


def test_kb_search_topk(kb_client):
    resp = kb_client.get("/api/kb/search", params={"q": "邮费是多少", "strategy": "dense"})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits and "score" in hits[0]
    joined = hits[0]["question"] + hits[0]["answer"]
    assert "99" in "".join(h["answer"] for h in hits)  # 运费说明命中


def test_kb_jobs_unknown_kind_rejected(kb_client):
    resp = kb_client.post("/api/kb/jobs/nope")
    assert resp.status_code == 422


def test_kb_materials_dryrun(kb_client):
    resp = kb_client.get("/api/kb/materials")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 3  # 三份 md + faq 表
    assert body["total_chunks"] > 0
    m = body["items"][0]
    assert {"file", "type", "chars", "lines", "chunks", "key_clauses", "features"} <= set(m)
    # 售后手册有表格 → 特性标注"有表"
    manual = next(i for i in body["items"] if "手册" in i["file"])
    assert "有表" in manual["features"]
    # faq 表也是材料
    assert any("faq" in i["file"] for i in body["items"])


def test_kb_mining_stats(kb_client):
    # A fresh demo database has no mining history: create and remove our own fixture.
    import uuid
    from sqlalchemy import delete
    from app.db import get_session_factory
    from app.knowledge.models import QaExtractionStaging

    factory = get_session_factory()
    with factory() as session:
        row = QaExtractionStaging(
            batch_no="mine_" + uuid.uuid4().hex[:14],
            source_ref="synthetic-api-test", question="合成挖掘测试问题",
            answer="合成测试答案", status="kept",
        )
        session.add(row)
        session.commit()
        row_id = row.id
    try:
        resp = kb_client.get("/api/kb/mining")
        assert resp.status_code == 200
        body = resp.json()
        assert {"extracted", "kept", "discarded", "runs", "recent"} <= set(body)
        assert body["kept"] >= 1
        assert isinstance(body["runs"], int) and body["runs"] >= 1
        assert any(item["id"] == row_id for item in body["recent"])
    finally:
        with factory() as session:
            session.execute(delete(QaExtractionStaging).where(QaExtractionStaging.id == row_id))
            session.commit()


def test_kb_dualwrite_ledger(kb_client):
    resp = kb_client.get("/api/kb/dualwrite")
    assert resp.status_code == 200
    body = resp.json()
    assert {"pending", "done", "milvus_count", "collection", "ledger"} <= set(body)
    assert body["done"] >= 1
    assert body["milvus_count"] >= 1  # Milvus 实数
    assert body["collection"] == "knowledge"
    if body["ledger"]:
        assert {"id", "questions", "vectorize_status", "vector_id"} <= set(body["ledger"][0])


def test_kb_preview_reports_duplicates(kb_client):
    # 预览一段与库内 FAQ 完全重复的内容 → 查重提示
    md = "# FAQ导入\n退货政策是什么\n云优选支持7天无理由退货。"
    resp = kb_client.post("/api/kb/preview", json={"markdown": md})
    assert resp.status_code == 200
    body = resp.json()
    assert "dup_count" in body
    assert body["dup_count"] >= 1  # "退货政策是什么"与库内重复
