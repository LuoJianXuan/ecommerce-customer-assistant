"""ch06 多轮指代消解+意图联合评估 runner。

逐组滚动历史：每轮先 aresolve（消解+改写）再 aclassify（意图），断言
① 意图判对；② 消解后问句包含标注关键词；③ exact 轮归一化后与原句一致（不强行改写）。

用法：
    python evals/ch06/run_coref_eval.py
    python evals/ch06/run_coref_eval.py --repeats 2

通过线（spec §7）：全部轮次通过率 ≥95%，且组名带「验收组」的场景 100%。
组内轮次串行（有依赖），组间并发（信号量限 5）。
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app.agent.coref import aresolve  # noqa: E402
from app.agent.intent import aclassify  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import build_intent_model  # noqa: E402

EVAL_FILE = Path(__file__).parent / "coref_cases.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"[\s，。！？!?.,、～「」]", "", str(s))


async def run_group(group: dict, llm) -> list[dict]:
    """组内逐轮滚动历史，返回每轮判定记录。"""
    history: list = []
    rows = []
    for turn in group["turns"]:
        resolved = await aresolve(llm, history, turn["u"])
        r = await aclassify(llm, resolved.resolved_query)
        intent_ok = r.intent == turn["intent"]
        kw_ok = all(k in resolved.resolved_query for k in turn.get("keywords", []))
        exact_ok = True
        if turn.get("exact"):
            exact_ok = _norm(resolved.resolved_query) == _norm(turn["u"])
        rows.append({
            "u": turn["u"], "resolved": resolved.resolved_query, "degraded": resolved.degraded,
            "expect": turn["intent"], "got": r.intent,
            "intent_ok": intent_ok, "kw_ok": kw_ok, "exact_ok": exact_ok,
            "ok": intent_ok and kw_ok and exact_ok,
        })
        history.extend([HumanMessage(content=turn["u"]), AIMessage(content=turn["a"])])
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    groups = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm = build_intent_model(settings)
    semaphore = asyncio.Semaphore(5)

    print(f"模型={settings.intent_model or settings.llm_model}  组数={len(groups)}  "
          f"总轮数={sum(len(g['turns']) for g in groups)}  repeats={args.repeats}")

    async def guarded(group):
        async with semaphore:
            return await run_group(group, llm)

    all_ok = True
    for run_idx in range(args.repeats):
        results = await asyncio.gather(*(guarded(g) for g in groups))
        total = passed = 0
        accept_total = accept_passed = 0
        print(f"{'用户':<22}{'期望':<10}{'实际':<10}{'消解结果':<34}标记")
        print("-" * 88)
        for group, rows in zip(groups, results):
            is_accept = "验收组" in group["group"]
            for row in rows:
                total += 1
                passed += row["ok"]
                if is_accept:
                    accept_total += 1
                    accept_passed += row["ok"]
                mark = "✓" if row["ok"] else "✗"
                flags = []
                if not row["intent_ok"]:
                    flags.append("意图错")
                if not row["kw_ok"]:
                    flags.append("关键词缺")
                if not row["exact_ok"]:
                    flags.append("被改写")
                if row["degraded"]:
                    flags.append("降级透传")
                print(f"{row['u'][:20]:<22}{row['expect']:<10}{row['got']:<10}"
                      f"{row['resolved'][:32]:<34}{mark}{'/'.join(flags)}")
        print("-" * 88)
        rate = passed / total
        ok = rate >= 0.95 and accept_passed == accept_total
        all_ok = all_ok and ok
        print(f"[run {run_idx + 1}] 轮次通过 {passed}/{total}({rate:.1%})  "
              f"验收组 {accept_passed}/{accept_total}  -> {'通过' if ok else '未通过'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
