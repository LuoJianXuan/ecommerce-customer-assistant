"""ch10 批量主题归类、分布聚合与明细查询服务。"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.models import LowConfidenceQuestion, TopicClassification
from app.topics.inference import LocalTopicClassifier, get_topic_classifier
from app.topics.taxonomy import TOPIC_NAMES


class UnknownQuestionIds(ValueError):
    def __init__(self, ids: list[int]):
        self.ids = ids
        super().__init__(f"低置信问题不存在:{ids}")


class TopicClassificationService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        classifier: LocalTopicClassifier | None = None,
    ):
        self.session_factory = session_factory
        self._classifier = classifier

    @property
    def classifier(self) -> LocalTopicClassifier:
        """只在 POST 批量归类时获取模型；分布与明细查询不依赖模型文件。"""
        if self._classifier is None:
            self._classifier = get_topic_classifier()
        return self._classifier

    def classify_batch(
        self,
        question_ids: list[int] | None = None,
        limit: int = 100,
        reclassify: bool = False,
    ) -> dict:
        batch_id = f"topic-{datetime.now():%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        with self.session_factory() as session:
            stmt = select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id)
            if question_ids:
                unique_ids = list(dict.fromkeys(question_ids))
                stmt = stmt.where(LowConfidenceQuestion.id.in_(unique_ids))
            else:
                stmt = stmt.limit(limit)
                if not reclassify:
                    stmt = (
                        select(LowConfidenceQuestion)
                        .outerjoin(
                            TopicClassification,
                            TopicClassification.question_id == LowConfidenceQuestion.id,
                        )
                        .where(TopicClassification.id.is_(None))
                        .order_by(LowConfidenceQuestion.id)
                        .limit(limit)
                    )
            # 默认攒批跳过其他工作器已锁定的行，两个批次会自然分片；显式 ID
            # 表示调用方点名这些问题，应等待前一事务完成再读取最新分类状态，
            # 否则 SKIP LOCKED 会把“正在处理”误报成“不存在”。
            locked_stmt = (
                stmt.with_for_update() if question_ids else stmt.with_for_update(skip_locked=True)
            )
            rows = list(session.scalars(locked_stmt).all())

            if question_ids:
                found = {row.id for row in rows}
                missing = sorted(set(question_ids) - found)
                if missing:
                    raise UnknownQuestionIds(missing)

            existing = {
                row.question_id: row
                for row in session.scalars(
                    select(TopicClassification).where(
                        TopicClassification.question_id.in_([r.id for r in rows] or [-1])
                    )
                ).all()
            }
            targets = rows if reclassify else [r for r in rows if r.id not in existing]
            predictions = (
                self.classifier.predict([r.raw_question for r in targets]) if targets else []
            )
            if len(predictions) != len(targets):
                raise RuntimeError(
                    f"主题模型返回 {len(predictions)} 条,预期 {len(targets)} 条"
                )
            for question, prediction in zip(targets, predictions):
                record = existing.get(question.id)
                if record is None:
                    record = TopicClassification(question_id=question.id)
                    session.add(record)
                record.labels = prediction.labels
                record.scores = prediction.scores
                record.model_version = self.classifier.model_version
                record.batch_id = batch_id
                record.classified_at = datetime.now()
            session.commit()

        return {
            "batch_id": batch_id,
            "requested": len(question_ids) if question_ids else limit,
            "selected": len(rows),
            "classified": len(targets),
            "skipped_already_classified": len(rows) - len(targets),
            "model_version": self.classifier.model_version,
            "threshold": self.classifier.threshold,
            "device": self.classifier.device if targets else None,
        }

    def pending_count(self, source: str | None = None) -> int:
        stmt = (
            select(func.count(LowConfidenceQuestion.id))
            .outerjoin(
                TopicClassification,
                TopicClassification.question_id == LowConfidenceQuestion.id,
            )
            .where(TopicClassification.id.is_(None))
        )
        if source:
            stmt = stmt.where(LowConfidenceQuestion.source == source)
        with self.session_factory() as session:
            return int(session.scalar(stmt) or 0)

    @staticmethod
    def _filters(stmt, source, batch_id, start, end):
        if source:
            stmt = stmt.where(LowConfidenceQuestion.source == source)
        if batch_id:
            stmt = stmt.where(TopicClassification.batch_id == batch_id)
        if start:
            stmt = stmt.where(TopicClassification.classified_at >= start)
        if end:
            stmt = stmt.where(TopicClassification.classified_at <= end)
        return stmt

    def distribution(
        self,
        source: str | None = None,
        batch_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict:
        stmt = (
            select(TopicClassification.labels)
            .join(
                LowConfidenceQuestion,
                LowConfidenceQuestion.id == TopicClassification.question_id,
            )
            .order_by(TopicClassification.id)
        )
        stmt = self._filters(stmt, source, batch_id, start, end)
        with self.session_factory() as session:
            rows = list(session.scalars(stmt).all())
        counts = {name: 0 for name in TOPIC_NAMES}
        unidentified = 0
        for labels in rows:
            if not labels:
                unidentified += 1
            for label in labels or []:
                if label in counts:
                    counts[label] += 1
        total = len(rows)
        items = [
            {
                "label": name,
                "count": counts[name],
                "question_ratio": round(counts[name] / total, 4) if total else 0.0,
            }
            for name in TOPIC_NAMES
        ]
        items.append(
            {
                "label": "未识别",
                "count": unidentified,
                "question_ratio": round(unidentified / total, 4) if total else 0.0,
            }
        )
        return {
            "classified_questions": total,
            # 待处理记录没有 batch/time 属性；仅在全量或来源视图返回，避免制造伪口径。
            "pending_questions": (
                self.pending_count(source) if not (batch_id or start or end) else None
            ),
            "label_assignments": sum(counts.values()),
            "items": items,
        }

    def items(
        self,
        label: str | None = None,
        source: str | None = None,
        batch_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        stmt = select(TopicClassification, LowConfidenceQuestion).join(
            LowConfidenceQuestion,
            LowConfidenceQuestion.id == TopicClassification.question_id,
        )
        stmt = self._filters(stmt, source, batch_id, start, end)
        if label:
            if label == "未识别":
                stmt = stmt.where(func.json_length(TopicClassification.labels) == 0)
            else:
                stmt = stmt.where(
                    func.json_contains(
                        TopicClassification.labels,
                        json.dumps(label, ensure_ascii=False),
                    )
                    == 1
                )
        count_stmt = select(func.count()).select_from(stmt.subquery())
        stmt = stmt.order_by(TopicClassification.classified_at.desc(), TopicClassification.id.desc())
        stmt = stmt.offset(offset).limit(limit)
        with self.session_factory() as session:
            total = int(session.scalar(count_stmt) or 0)
            rows = session.execute(stmt).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "id": classification.id,
                    "question_id": question.id,
                    "raw_question": question.raw_question,
                    "source": question.source,
                    "reason": question.reason,
                    "labels": classification.labels,
                    "scores": classification.scores,
                    "model_version": classification.model_version,
                    "batch_id": classification.batch_id,
                    "classified_at": classification.classified_at.isoformat(),
                }
                for classification, question in rows
            ],
        }
