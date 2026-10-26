"""ch09 统一评估流水线：复用 ch04 指标，保存报告并记录趋势。"""

import json
from datetime import datetime

from app.config import get_settings
from app.db import get_session_factory
from app.knowledge.models import EvalRun


def summarize_metrics(retrieval: dict, generation: dict) -> dict:
    chosen = retrieval["hybrid_rerank"]["ALL"]
    judged = generation.get("judged", 0)
    refused = generation.get("refused_correct", 0) + generation.get("refused_wrong", 0)
    return {
        "retrieval": {
            "recall_at_5": chosen["r5"],
            "recall_at_10": chosen["r10"],
            "mrr": chosen["mrr"],
            "buckets": retrieval["hybrid_rerank"],
        },
        "generation": {
            **generation,
            "faithfulness": 1 - generation.get("fabricated", 0) / judged if judged else 0,
            "refusal_rate": generation.get("refused_correct", 0) / refused if refused else 0,
        },
    }


async def run_pipeline(trigger: str = "manual", progress=None) -> int:
    """执行一次全量检索评估和限样生成评估，返回 eval_runs.id。"""
    import evals.ch04.run_eval as ev
    from app.llm import build_chat_model

    settings = get_settings()
    cases = ev.load_cases()
    gt_ids = ev.resolve_gt_ids(cases)
    rewrite_llm = build_chat_model(settings) if settings.query_rewrite_enabled else None
    if progress:
        progress(f"[检索段] {len(cases)} 题开始")
    retrieval = ev.eval_retrieval(cases, gt_ids, rewrite_llm, progress=progress)
    if progress:
        progress(f"[生成段] 限样 {settings.eval_gen_limit or '全量'}")
    generation = await ev.eval_generation(
        cases,
        progress=progress,
        gen_limit=settings.eval_gen_limit or None,
    )

    now = datetime.now()
    run_ts = now.strftime("%Y-%m-%d %H:%M")
    payload = {
        "run_ts": run_ts,
        "run_at": now.isoformat(),
        "cases_total": len(cases),
        "retrieval": retrieval,
        "generation": generation,
    }
    ev.REPORT_MD.write_text(ev.fmt_report(retrieval, generation, run_ts), encoding="utf-8")
    ev.REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    factory = get_session_factory()
    with factory() as session:
        row = EvalRun(
            trigger=trigger,
            case_count=len(cases),
            metrics=summarize_metrics(retrieval, generation),
        )
        session.add(row)
        session.commit()
        return row.id
