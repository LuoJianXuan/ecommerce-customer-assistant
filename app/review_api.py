"""ch09 待审知识缺口 API。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.db import get_session_factory
from app.knowledge.flywheel import publish_review_item
from app.knowledge.models import LowConfidenceQuestion, ReviewQueue

router = APIRouter(prefix="/api/review", tags=["review"])


class ApproveRequest(BaseModel):
    approved_answer: str


def _item(row: ReviewQueue) -> dict:
    return {
        "id": row.id,
        "std_question": row.std_question,
        "sample_answer": row.sample_answer,
        "occurrence_count": row.occurrence_count,
        "status": row.status,
        "approved_answer": row.approved_answer,
        "created_at": row.created_at.strftime("%m-%d %H:%M") if row.created_at else "",
        "updated_at": row.updated_at.strftime("%m-%d %H:%M") if row.updated_at else "",
    }


@router.get("/items")
def list_items(status: str = "待审") -> dict:
    factory = get_session_factory()
    with factory() as session:
        stmt = select(ReviewQueue).order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.id)
        if status:
            stmt = stmt.where(ReviewQueue.status == status)
        rows = session.scalars(stmt).all()
    return {"items": [_item(row) for row in rows]}


@router.get("/items/{review_id}")
def review_detail(review_id: int) -> dict:
    factory = get_session_factory()
    with factory() as session:
        row = session.get(ReviewQueue, review_id)
        if row is None:
            raise HTTPException(404, f"待审问题不存在: {review_id}")
        raw_rows = session.scalars(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.created_at, LowConfidenceQuestion.id)
        ).all()
        result = _item(row)
        result["raw_questions"] = [
            {
                "id": item.id,
                "raw_question": item.raw_question,
                "source": item.source,
                "reason": item.reason,
                "retrieval_snapshot": item.retrieval_snapshot or [],
                "created_at": item.created_at.strftime("%m-%d %H:%M") if item.created_at else "",
            }
            for item in raw_rows
        ]
    return result


@router.post("/items/{review_id}/approve")
def approve_item(review_id: int, body: ApproveRequest) -> dict:
    answer = body.approved_answer.strip()
    if not answer:
        raise HTTPException(422, "审核答案不能为空")
    try:
        chunk_id = publish_review_item(review_id, answer)
    except LookupError:
        raise HTTPException(404, f"待审问题不存在: {review_id}") from None
    except ValueError as exc:
        raise HTTPException(409, f"问题已经处理: {exc}") from None
    return {"id": review_id, "status": "已通过", "chunk_id": chunk_id}


@router.post("/items/{review_id}/reject")
def reject_item(review_id: int) -> dict:
    factory = get_session_factory()
    with factory() as session:
        row = session.scalar(
            select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update()
        )
        if row is None:
            raise HTTPException(404, f"待审问题不存在: {review_id}")
        if row.status != "待审":
            raise HTTPException(409, f"问题已经处理: {row.status}")
        row.status = "已驳回"
        session.commit()
    return {"id": review_id, "status": "已驳回"}
