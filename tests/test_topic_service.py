"""ch10 主题持久化集成测试；真实 MySQL + 假分类器。"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

from sqlalchemy import select

from app.db import get_session_factory
from app.knowledge.models import LowConfidenceQuestion, TopicClassification
from app.topics.inference import TopicPrediction
from app.topics.service import TopicClassificationService
from app.topics.taxonomy import TOPIC_NAMES


class FakeClassifier:
    model_version = "fake-v1"
    threshold = 0.4
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def predict(self, texts):
        self.calls += 1
        results = []
        for index, _ in enumerate(texts):
            labels = ["物流"] if index == 0 else ["物流", "运费"]
            scores = {name: (0.9 if name in labels else 0.01) for name in TOPIC_NAMES}
            results.append(TopicPrediction(labels=labels, scores=scores))
        return results


class BlockingClassifier(FakeClassifier):
    def __init__(self):
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def predict(self, texts):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocking classifier")
        return super().predict(texts)


def test_classify_batch_is_idempotent_and_queryable(require_db):
    from scripts.apply_ch10_schema import main as apply_schema

    apply_schema()
    factory = get_session_factory()
    marker = uuid4().hex
    with factory() as session:
        questions = [
            LowConfidenceQuestion(
                raw_question=f"{marker}-我的包裹到哪了",
                source="self_check",
                reason="test",
            ),
            LowConfidenceQuestion(
                raw_question=f"{marker}-快递和运费分别是多少",
                source="retrieval_low_conf",
                reason="test",
            ),
        ]
        session.add_all(questions)
        session.commit()
        ids = [question.id for question in questions]

    classifier = FakeClassifier()
    service = TopicClassificationService(factory, classifier)
    try:
        first = service.classify_batch(ids, reclassify=False)
        second = service.classify_batch(ids, reclassify=False)
        refreshed = service.classify_batch(ids, reclassify=True)

        assert first["classified"] == 2
        assert second["classified"] == 0
        assert second["skipped_already_classified"] == 2
        assert refreshed["classified"] == 2
        assert classifier.calls == 2

        distribution = service.distribution(batch_id=refreshed["batch_id"])
        assert distribution["classified_questions"] == 2
        assert distribution["pending_questions"] is None
        assert next(x for x in distribution["items"] if x["label"] == "物流")["count"] == 2

        items = service.items(label="运费", batch_id=refreshed["batch_id"])
        assert items["total"] == 1
        assert items["items"][0]["model_version"] == "fake-v1"
    finally:
        with factory.begin() as session:
            rows = list(
                session.scalars(
                    select(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_(ids))
                ).all()
            )
            for row in rows:
                session.delete(row)


def test_read_queries_do_not_initialize_classifier(require_db, monkeypatch):
    factory = get_session_factory()
    service = TopicClassificationService(factory)

    def fail_if_loaded():
        raise AssertionError("read-only query must not initialize model")

    monkeypatch.setattr("app.topics.service.get_topic_classifier", fail_if_loaded)
    assert "items" in service.distribution()
    assert "items" in service.items(limit=1)


def test_concurrent_explicit_batches_wait_and_remain_idempotent(require_db):
    from scripts.apply_ch10_schema import main as apply_schema

    apply_schema()
    factory = get_session_factory()
    marker = uuid4().hex
    with factory() as session:
        question = LowConfidenceQuestion(
            raw_question=f"{marker}-并发归类测试",
            source="self_check",
            reason="test",
        )
        session.add(question)
        session.commit()
        question_id = question.id

    first_classifier = BlockingClassifier()
    second_classifier = FakeClassifier()
    first_service = TopicClassificationService(factory, first_classifier)
    second_service = TopicClassificationService(factory, second_classifier)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(first_service.classify_batch, [question_id])
            assert first_classifier.entered.wait(timeout=3)
            second = pool.submit(second_service.classify_batch, [question_id])
            time.sleep(0.2)
            assert not second.done(), "显式 ID 的后发批次应等待行锁，而不是误报不存在"
            first_classifier.release.set()
            results = [first.result(timeout=5), second.result(timeout=5)]

        assert sorted(result["classified"] for result in results) == [0, 1]
        assert first_classifier.calls == 1
        assert second_classifier.calls == 0
    finally:
        first_classifier.release.set()
        with factory.begin() as session:
            row = session.get(LowConfidenceQuestion, question_id)
            if row is not None:
                session.delete(row)
