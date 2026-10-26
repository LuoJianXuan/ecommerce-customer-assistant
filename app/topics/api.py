"""ch10 主题分类旁路 API。"""

from datetime import datetime
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.db import get_session_factory
from app.topics.service import (
    TopicClassificationService,
    UnknownQuestionIds,
)
from app.topics.taxonomy import TOPIC_NAMES

router = APIRouter(prefix="/api/topics", tags=["topics"])


class ClassifyBatchRequest(BaseModel):
    question_ids: list[int] | None = Field(default=None, min_length=1, max_length=500)
    limit: int = Field(default=100, ge=1, le=500)
    reclassify: bool = False


@lru_cache
def get_topic_service() -> TopicClassificationService:
    return TopicClassificationService(get_session_factory())


SourceFilter = Literal["retrieval_low_conf", "self_check", "user_feedback"]


@router.post("/classify-batch")
def classify_batch(
    body: ClassifyBatchRequest,
    service: TopicClassificationService = Depends(get_topic_service),
) -> dict:
    try:
        return service.classify_batch(body.question_ids, body.limit, body.reclassify)
    except UnknownQuestionIds as exc:
        raise HTTPException(404, detail={"message": str(exc), "missing_ids": exc.ids}) from None
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(503, f"主题分类服务不可用:{exc}") from None


@router.get("/distribution")
def distribution(
    source: SourceFilter | None = None,
    batch_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    service: TopicClassificationService = Depends(get_topic_service),
) -> dict:
    if start and end and start > end:
        raise HTTPException(422, "start 不能晚于 end")
    return service.distribution(source, batch_id, start, end)


@router.get("/items")
def items(
    label: str | None = None,
    source: SourceFilter | None = None,
    batch_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: TopicClassificationService = Depends(get_topic_service),
) -> dict:
    if label and label not in {*TOPIC_NAMES, "未识别"}:
        raise HTTPException(422, f"未知主题标签:{label}")
    if start and end and start > end:
        raise HTTPException(422, "start 不能晚于 end")
    return service.items(label, source, batch_id, start, end, limit, offset)
