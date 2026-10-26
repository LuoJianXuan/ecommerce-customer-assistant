"""ch09 对照实验：RERANK_CANDIDATES=20 只跑 hybrid_rerank 检索段，归因指标口径差异。

用法：RERANK_CANDIDATES=20 python evals/ch09/retrieval_candidates20.py
（进程 env 优先级高于 .env——pydantic-settings 顺序；脚本断言实际生效值防"假跑"。
  只做归因，不写 eval_runs——趋势表保持 6 候选运行口径纯净；结果落
  evals/ch09/candidates20_result.json，对比结论记 dev-notes。）
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import evals.ch04.run_eval as ev  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    assert settings.rerank_candidates == 20, (
        f"RERANK_CANDIDATES 未生效: {settings.rerank_candidates}（须以进程 env=20 启动）"
    )
    cases = ev.load_cases()
    gt_ids = ev.resolve_gt_ids(cases)
    ev.STRATEGIES = ["hybrid_rerank"]  # 只跑对照需要的这一策略
    rewrite_llm = None
    if settings.query_rewrite_enabled:
        from app.llm import build_chat_model

        rewrite_llm = build_chat_model(settings)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    retrieval = asyncio.run(ev_retrieval_async(cases, gt_ids, rewrite_llm, progress))
    payload = {
        "run_ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rerank_candidates": settings.rerank_candidates,
        "rerank_top_k": settings.rerank_top_k,
        "query_rewrite_enabled": settings.query_rewrite_enabled,
        "hybrid_rerank": retrieval["hybrid_rerank"],
    }
    out = Path(__file__).parent / "candidates20_result.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写入", out)
    print(json.dumps(retrieval["hybrid_rerank"]["ALL"], ensure_ascii=False))
    return 0


async def ev_retrieval_async(cases, gt_ids, rewrite_llm, progress):
    return ev.eval_retrieval(cases, gt_ids, rewrite_llm, progress=progress)


if __name__ == "__main__":
    raise SystemExit(main())
