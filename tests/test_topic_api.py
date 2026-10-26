"""ch10 主题 API 测试；分类模型使用假服务，不加载本地权重。"""

from fastapi.testclient import TestClient

from app.main import create_app
from app.topics.api import get_topic_service
from app.topics.service import UnknownQuestionIds


class FakeTopicService:
    def __init__(self):
        self.calls = []

    def classify_batch(self, question_ids, limit, reclassify):
        self.calls.append((question_ids, limit, reclassify))
        if question_ids == [999]:
            raise UnknownQuestionIds([999])
        return {
            "batch_id": "topic-test",
            "requested": len(question_ids) if question_ids else limit,
            "selected": 2,
            "classified": 2,
            "skipped_already_classified": 0,
            "model_version": "fake-v1",
            "threshold": 0.4,
            "device": "cpu",
        }

    def distribution(self, source, batch_id, start, end):
        self.calls.append(("distribution", source, batch_id, start, end))
        return {
            "classified_questions": 2,
            "pending_questions": 1,
            "label_assignments": 3,
            "items": [],
        }

    def items(self, label, source, batch_id, start, end, limit, offset):
        self.calls.append(("items", label, source, batch_id, start, end, limit, offset))
        return {"total": 0, "limit": limit, "offset": offset, "items": []}


def make_topic_client():
    app = create_app()
    service = FakeTopicService()
    app.dependency_overrides[get_topic_service] = lambda: service
    return TestClient(app), service


def test_batch_classification_endpoint_uses_fake_service():
    client, service = make_topic_client()

    response = client.post(
        "/api/topics/classify-batch",
        json={"question_ids": [3, 7], "limit": 20, "reclassify": True},
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "fake-v1"
    assert service.calls == [([3, 7], 20, True)]


def test_batch_classification_returns_404_for_missing_question():
    client, _ = make_topic_client()
    response = client.post("/api/topics/classify-batch", json={"question_ids": [999]})
    assert response.status_code == 404
    assert response.json()["detail"]["missing_ids"] == [999]


def test_batch_classification_rejects_explicit_empty_selection():
    client, service = make_topic_client()
    response = client.post("/api/topics/classify-batch", json={"question_ids": []})
    assert response.status_code == 422
    assert service.calls == []


def test_distribution_endpoint_forwards_filters():
    client, service = make_topic_client()
    response = client.get(
        "/api/topics/distribution",
        params={"source": "self_check", "batch_id": "topic-test"},
    )
    assert response.status_code == 200
    assert response.json()["classified_questions"] == 2
    assert service.calls[0][0:3] == ("distribution", "self_check", "topic-test")


def test_distribution_rejects_unknown_source():
    client, service = make_topic_client()
    response = client.get("/api/topics/distribution", params={"source": "typo"})
    assert response.status_code == 422
    assert service.calls == []


def test_items_rejects_unknown_label_before_service_call():
    client, service = make_topic_client()
    response = client.get("/api/topics/items", params={"label": "不存在的类目"})
    assert response.status_code == 422
    assert service.calls == []


def test_items_endpoint_supports_pagination():
    client, service = make_topic_client()
    response = client.get(
        "/api/topics/items", params={"label": "物流", "limit": 20, "offset": 40}
    )
    assert response.status_code == 200
    assert response.json()["offset"] == 40
    assert service.calls[0][0:3] == ("items", "物流", None)


def test_topic_page_and_management_pages_are_linked():
    client, _ = make_topic_client()
    page = client.get("/topic")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "低置信问题主题分布" in page.text
    assert 'id="classifyBtn"' in page.text
    assert 'id="chart"' in page.text

    for path in ("/", "/kb", "/review", "/eval"):
        linked = client.get(path)
        assert linked.status_code == 200
        assert 'href="/topic"' in linked.text
