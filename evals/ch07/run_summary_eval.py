"""ch07 摘要评估：要点覆盖 100% + 零编造（纯 Prompt 任务，评估集替代 TDD）。

合格标准（教程原话）：梗概能让下游读出「聊过哪款耳机、订单号多少、卡在哪一步」。

用法：python evals/ch07/run_summary_eval.py
通过线：全部用例要点覆盖 100%；编造检测（摘要中的订单号/数字串必须源于原文）零命中。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import get_settings  # noqa: E402
from app.context_summary import fabrication_check, summarize_batch  # noqa: E402
from app.llm import build_summary_model  # noqa: E402

EVAL_FILE = Path(__file__).parent / "summary_cases.jsonl"


async def main() -> int:
    settings = get_settings()
    llm = build_summary_model(settings)
    cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"模型={settings.llm_model}(t=0)  用例数={len(cases)}")
    print(f"{'用例':<10}{'要点覆盖':<12}{'编造':<10}梗概")
    print("-" * 90)
    all_ok = True
    for case in cases:
        dialog_text = "\n".join(f"{role}：{text}" for role, text in case["dialog"])
        summary = await summarize_batch(llm, dialog_text)
        missing = [f for f in case["facts"] if f not in summary]
        fabricated = fabrication_check(summary, dialog_text)
        ok = not missing and not fabricated and len(summary) <= 250
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗"
        note = (f" 缺{missing}" if missing else "") + (f" 编造{fabricated}" if fabricated else "") + (" 超长" if len(summary) > 250 else "")
        print(f"{case['id']:<10}{len(case['facts']) - len(missing)}/{len(case['facts']):<10}{len(fabricated):<10}{mark}{note} {summary[:52]}")
    print("-" * 90)
    print("-> " + ("通过（要点覆盖 100%、编造 0）" if all_ok else "未通过"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
