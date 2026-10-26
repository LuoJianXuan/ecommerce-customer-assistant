"""/api/eval/* 评估与编造个案台账接口。

- POST /api/eval/run：就地重跑（后台任务 + 进度日志行，页面终端回显数据源）
- GET  /api/eval/report：最新 report.json
- GET  /api/eval/cases：台账列表 + 两率聚合（判出率/确认幻觉率，口径见文档）
- POST /api/eval/cases/{eval_id}/resolve：处置（已解决/无需解决，resolution 必填）

两率口径（教程文档原文）：
- 判出率 = (本轮裁判判出的编造个案 + 库外该拒没拒) ÷ 实际评上的题数
- 确认幻觉率 = 人工点成「已解决」的个案 ÷ 同分母（下界：无需解决不算、未过目不算）
- 本轮 = last_seen_at >= 最新评估 run 的时间
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import get_session_factory
from app.eval_run_lock import claim_eval_run, release_eval_run
from app.knowledge.models import EvalRun, FaithCase

REPORT_JSON = Path(__file__).resolve().parent.parent / "evals" / "ch04" / "report.json"

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

_jobs: dict[str, dict] = {}
_logs: dict[str, list[str]] = {}
_lock = threading.Lock()
_current_run: str | None = None
TREND_KEYS = ("recall_at_5", "recall_at_10", "mrr", "faithfulness", "refusal_rate")


def _log(job_id: str, line: str) -> None:
    with _lock:
        _logs.setdefault(job_id, []).append(f"[{datetime.now():%H:%M:%S}] {line}")


def _trend_metrics(metrics: dict) -> dict:
    if "retrieval" not in metrics:
        return {key: metrics[key] for key in TREND_KEYS if key in metrics}
    retrieval = metrics.get("retrieval") or {}
    generation = metrics.get("generation") or {}
    return {
        key: (retrieval if key in retrieval else generation)[key]
        for key in TREND_KEYS
        if key in retrieval or key in generation
    }


@router.post("/run")
def run_eval() -> dict:
    global _current_run
    with _lock:
        if _current_run or not claim_eval_run():
            raise HTTPException(409, "评估已在运行")
        job_id = uuid.uuid4().hex[:8]
        _jobs[job_id] = {"id": job_id, "label": "评估重跑（四策略×100题）", "status": "running"}
        _logs[job_id] = []
        _current_run = job_id

    def runner() -> None:
        import asyncio

        try:
            from evals.ch09.run_pipeline import run_pipeline

            run_id = asyncio.run(run_pipeline("manual", progress=lambda l: _log(job_id, l)))
            _log(job_id, f"✓ 完成：评估轮次 #{run_id} 已保存")
            with _lock:
                _jobs[job_id]["status"] = "done"
        except Exception as exc:  # 后台任务失败要可见
            _log(job_id, f"✗ 失败: {type(exc).__name__}: {exc}")
            with _lock:
                _jobs[job_id]["status"] = "failed"
        finally:
            global _current_run
            with _lock:
                _jobs[job_id]["result"] = "见日志"
                _current_run = None
            release_eval_run()

    threading.Thread(target=runner, daemon=True).start()
    return _jobs[job_id]


@router.get("/jobs")
def job_status() -> dict:
    with _lock:
        items = [{"id": j["id"], "label": j["label"], "status": j["status"]} for j in _jobs.values()]
    return {"items": items}


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, after: int = 0) -> dict:
    with _lock:
        lines = _logs.get(job_id, [])
    return {"lines": lines[after:], "total": len(lines)}


@router.get("/report")
def report() -> dict:
    if not REPORT_JSON.exists():
        return {"exists": False}
    return {"exists": True, **json.loads(REPORT_JSON.read_text(encoding="utf-8"))}


@router.get("/runs")
def eval_runs() -> dict:
    factory = get_session_factory()
    with factory() as session:
        rows = session.scalars(
            select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
        ).all()
    items = []
    for index, row in enumerate(rows):
        current = _trend_metrics(row.metrics)
        previous = _trend_metrics(rows[index + 1].metrics) if index + 1 < len(rows) else {}
        delta = {
            key: round(value - previous[key], 6)
            for key, value in current.items()
            if key in previous and isinstance(value, (int, float))
            and isinstance(previous[key], (int, float))
        }
        items.append({
            "id": row.id,
            "trigger": row.trigger,
            "case_count": row.case_count,
            "metrics": row.metrics,
            "delta": delta,
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else "",
        })
    return {"items": items}


def _rates() -> dict:
    """两率聚合（口径见模块 docstring）。"""
    factory = get_session_factory()
    if not REPORT_JSON.exists():
        return {}
    run_at = json.loads(REPORT_JSON.read_text(encoding="utf-8"))["run_at"]
    run_dt = datetime.fromisoformat(run_at)
    gen = json.loads(REPORT_JSON.read_text(encoding="utf-8"))["generation"]
    judged = gen.get("judged", 0) + gen.get("refused_correct", 0) + gen.get("refused_wrong", 0)
    if not judged:
        return {}
    with factory() as s:
        rows = s.scalars(select(FaithCase)).all()
    this_round = [r for r in rows if r.last_seen_at and r.last_seen_at >= run_dt]
    fabricated_now = sum(1 for r in this_round if r.bucket != "D_absent") + gen.get("refused_wrong", 0)
    confirmed = sum(1 for r in this_round if r.status == "已解决")
    return {
        "judged": judged,
        "detected": fabricated_now,
        "confirmed": confirmed,
        "detect_rate": fabricated_now / judged,
        "confirm_rate": confirmed / judged,
        "cross_round_total": len(rows),
    }


@router.get("/cases")
def cases(status: str = "") -> dict:
    factory = get_session_factory()
    with factory() as s:
        stmt = select(FaithCase).order_by(FaithCase.last_seen_at.desc())
        if status:
            stmt = stmt.where(FaithCase.status == status)
        rows = s.scalars(stmt).all()
    return {
        "rates": _rates(),
        "items": [
            {
                "eval_id": r.eval_id, "bucket": r.bucket, "query": r.query,
                "answer": r.answer, "reason": r.reason, "citations": r.citations or [],
                "judge_model": r.judge_model, "status": r.status,
                "seen_count": r.seen_count, "resolution": r.resolution,
                "first_seen_at": r.first_seen_at.strftime("%m-%d %H:%M") if r.first_seen_at else "",
                "last_seen_at": r.last_seen_at.strftime("%m-%d %H:%M") if r.last_seen_at else "",
            }
            for r in rows
        ],
    }


class ResolveRequest(BaseModel):
    status: str = Field(pattern="^(已解决|无需解决)$")
    resolution: str = Field(min_length=5, max_length=300, description="处置必须留交代：怎么解决的/为什么不用改")


@router.post("/cases/{eval_id}/resolve")
def resolve(eval_id: str, req: ResolveRequest) -> dict:
    factory = get_session_factory()
    with factory() as s:
        row = s.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
        if row is None:
            raise HTTPException(404, f"个案不存在: {eval_id}")
        row.status = req.status
        row.resolution = req.resolution
        row.resolved_at = datetime.now()
        s.commit()
    return {"eval_id": eval_id, "status": req.status}
