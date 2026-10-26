"""检索评估集：向语义检索命中率跑分（替代模型行为类 TDD，ch03 工作要求）。

cases：ch02 关键词漏召回案例全集 + 验收锚点。
判定：期望 snippet 出现在 Top-K 任一命中块的 (question+answer) 里。
用法：python evals/retrieval_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.knowledge.retrieval import retrieve  # noqa: E402

EVAL_FILE = Path(__file__).parent / "retrieval_cases.jsonl"


async def main() -> int:
    cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = get_settings()
    hit = 0
    print(f"{'查询':<16} {'期望命中内容':<28} 判定  Top-1 相似度")
    print("-" * 78)
    for case in cases:
        from app.db import get_session_factory

        matches = retrieve(
            get_session_factory(), case["query"], top_k=settings.retrieval_top_k
        )
        found = any(
            case["expect_snippet"] in (m["question"] + m["answer"]) for m in matches
        )
        hit += found
        top1 = f"{matches[0]['score']:.3f}" if matches else "-"
        print(
            f"{case['query']:<16} {case['expect_snippet'][:14]:<28} "
            f"{'✅' if found else '❌'}   {top1}"
        )
    total = len(cases)
    print("-" * 78)
    print(f"Top-{settings.retrieval_top_k} 命中率: {hit}/{total} = {hit / total:.0%}")
    return 0 if hit == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
