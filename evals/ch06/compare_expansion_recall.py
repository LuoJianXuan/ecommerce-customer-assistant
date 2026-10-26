"""ch06 Query 扩写收益验证（spec §7）：扩写+多路检索 vs 单路检索的政策条款召回对比。

对若干退款/售后问题，各跑一遍：
- 单路：retrieve(strategy="hybrid_rerank", top_k=10)
- 多路：aexpand 真扩写 → retrieve_multi_query(final_k=10)
统计两路结果里政策类目（退货政策/售后手册）命中条数。通过线：多路政策命中 ≥ 单路。

用法：python evals/ch06/compare_expansion_recall.py   （需 Milvus/MySQL/LLM 在线）
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.query_expand import aexpand  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_session_factory  # noqa: E402
from app.knowledge.retrieval import retrieve, retrieve_multi_query  # noqa: E402
from app.llm import build_intent_model  # noqa: E402

POLICY_CATEGORIES = {"退货政策", "售后手册"}

QUESTIONS = [
    "这个能退吗",
    "还没收到货能退款吗",
    "退货的运费谁承担",
    "用了三天发现质量问题怎么换",
    "买的电饭煲想退货流程是什么",
    "退款多久能到账",
]


def policy_hits(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("category") in POLICY_CATEGORIES)


async def main() -> int:
    settings = get_settings()
    factory = get_session_factory()
    llm = build_intent_model(settings)
    print(f"模型={settings.intent_model or settings.llm_model}  问题数={len(QUESTIONS)}")
    print(f"{'问题':<24}{'单路政策命中':<12}{'多路政策命中':<12}{'扩写条数':<10}判定")
    print("-" * 70)
    total_single = total_multi = ok_count = 0
    for q in QUESTIONS:
        single = retrieve(factory, q, top_k=10, strategy="hybrid_rerank")
        queries = await aexpand(llm, q)
        multi = retrieve_multi_query(factory, queries, final_k=10)
        s, m = policy_hits(single), policy_hits(multi)
        total_single += s
        total_multi += m
        ok = m >= s
        ok_count += ok
        print(f"{q:<22}{s:<14}{m:<14}{len(queries):<12}{'✓' if ok else '✗ 多路更低'}")
    print("-" * 70)
    passed = total_multi >= total_single
    print(f"合计:单路 {total_single} 条 vs 多路 {total_multi} 条;"
          f"逐题多路≥单路 {ok_count}/{len(QUESTIONS)}  -> {'通过' if passed else '未通过'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
