"""ch09 自动评估趋势的公共 API 与指标收敛。"""

import pytest
from sqlalchemy import delete
from types import SimpleNamespace

from app.db import get_session_factory
from app.knowledge.models import EvalRun
from app.eval_run_lock import claim_eval_run, release_eval_run
from app.eval_scheduler import start_eval_scheduler
from evals.ch09.calibrate_gate import best_threshold
from evals.ch09.run_pipeline import summarize_metrics


def test_summarize_metrics_uses_hybrid_rerank_and_generation_rates():
    retrieval = {
        "hybrid_rerank": {
            "A_policy": {"r5": 0.7, "r10": 0.8, "mrr": 0.6},
            "ALL": {"r5": 0.8, "r10": 0.9, "mrr": 0.7},
        }
    }
    generation = {
        "judged": 8,
        "fabricated": 2,
        "refused_correct": 3,
        "refused_wrong": 1,
        "by_bucket": {"A_policy": {"judged": 4, "fabricated": 1}},
    }

    assert summarize_metrics(retrieval, generation) == {
        "retrieval": {
            "recall_at_5": 0.8,
            "recall_at_10": 0.9,
            "mrr": 0.7,
            "buckets": retrieval["hybrid_rerank"],
        },
        "generation": {
            **generation,
            "faithfulness": 0.75,
            "refusal_rate": 0.75,
        },
    }


def test_eval_runs_returns_latest_first_with_delta(client, require_db):
    factory = get_session_factory()
    with factory() as session:
        older = EvalRun(
            trigger="manual",
            case_count=100,
            metrics={"recall_at_5": 0.7, "faithfulness": 0.8},
        )
        newer = EvalRun(
            trigger="scheduled",
            case_count=100,
            metrics={
                "retrieval": {"recall_at_5": 0.75},
                "generation": {"faithfulness": 0.78},
            },
        )
        session.add_all([older, newer])
        session.commit()
        ids = [older.id, newer.id]

    try:
        resp = client.get("/api/eval/runs")
        assert resp.status_code == 200
        items = [item for item in resp.json()["items"] if item["id"] in ids]
        assert [item["id"] for item in items] == list(reversed(ids))
        assert items[0]["delta"] == {
            "recall_at_5": 0.05,
            "faithfulness": -0.02,
        }
    finally:
        with factory() as session:
            session.execute(delete(EvalRun).where(EvalRun.id.in_(ids)))
            session.commit()


def test_manual_and_scheduled_evaluations_share_one_lock():
    assert claim_eval_run()
    try:
        assert not claim_eval_run()
    finally:
        release_eval_run()
    assert claim_eval_run()
    release_eval_run()


def test_gate_threshold_balances_answerable_recall_and_correct_refusal():
    samples = [
        {"confidence": 0.5, "answerable": True},
        *({"confidence": 0.6, "answerable": False} for _ in range(4)),
        *({"confidence": 0.1, "answerable": False} for _ in range(5)),
    ]

    result = best_threshold(samples)

    assert result["threshold"] == 0.5
    assert result["f1"] > 0.7
    assert result["accuracy"] == 0.6


def test_gate_threshold_requires_both_sample_classes():
    with pytest.raises(ValueError, match="可回答和库外"):
        best_threshold([{"confidence": 0.5, "answerable": True}])


def test_zero_disables_daily_evaluation(monkeypatch):
    monkeypatch.setattr(
        "app.eval_scheduler.get_settings",
        lambda: SimpleNamespace(eval_daily_hour=0),
    )

    assert start_eval_scheduler() is None
