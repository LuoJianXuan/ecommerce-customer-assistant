"""ch06 意图评估 runner：意图识别正式版（四件套 prompt）对标注集跑分。

用法：
    python evals/ch06/run_intent_eval.py
    python evals/ch06/run_intent_eval.py --repeats 3            # 看稳定性分布
    INTENT_ESCALATION_ENABLED=true python evals/ch06/run_intent_eval.py  # 两级链路对照

通过线（spec §7）：总准确率 ≥95%，「其他」类样例 100% 落「其他」（怪问题不硬塞
业务意图），JSON 解析失败（重试后仍失败）次数为 0。
用例间并发（信号量限 5，对齐 run_routing_eval 的先例）。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.intent import FALLBACK_INTENT, aclassify  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import build_intent_model  # noqa: E402

EVAL_FILE = Path(__file__).parent / "intent_cases.jsonl"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    big = build_intent_model(settings)
    small = build_intent_model(settings, small=True) if settings.intent_escalation_enabled else None
    semaphore = asyncio.Semaphore(5)

    print(
        f"模型={settings.intent_model or settings.llm_model}  "
        f"升级链路={'开(小=' + settings.intent_small_model + ')' if small else '关'}  "
        f"阈值={settings.intent_confidence_threshold}  条数={len(cases)}  repeats={args.repeats}"
    )

    async def one(case):
        async with semaphore:
            return await aclassify(
                big, case["query"], small_llm=small, threshold=settings.intent_confidence_threshold
            )

    all_ok = True
    for run_idx in range(args.repeats):
        results = await asyncio.gather(*(one(c) for c in cases))
        hits = sum(r.intent == c["intent"] for r, c in zip(results, cases))
        others = [(r, c) for r, c in zip(results, cases) if c["intent"] == FALLBACK_INTENT]
        other_hits = sum(r.intent == FALLBACK_INTENT for r, _ in others)
        parse_fails = sum(r.parse_failed for r in results)
        escalated = sum(r.escalated for r in results)

        print(f"{'问句':<30}{'期望':<10}{'实际':<10}{'置信':<10}标记")
        print("-" * 72)
        for r, c in zip(results, cases):
            mark = "✓" if r.intent == c["intent"] else "✗"
            flag = (" 解析失败" if r.parse_failed else "") + (" 升级" if r.escalated else "")
            print(f"{c['query'][:28]:<30}{c['intent']:<10}{r.intent:<10}{r.confidence:<10.2f}{mark}{flag}")
        print("-" * 72)
        acc = hits / len(cases)
        ok = acc >= 0.95 and other_hits == len(others) and parse_fails == 0
        all_ok = all_ok and ok
        print(
            f"[run {run_idx + 1}] 准确率 {acc:.1%}({hits}/{len(cases)})  "
            f"其他类 {other_hits}/{len(others)}  解析失败 {parse_fails}  升级 {escalated}  "
            f"-> {'通过' if ok else '未通过'}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
