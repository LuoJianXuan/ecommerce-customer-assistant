"""用 ch04 100 题校准组合证据置信度阈值。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.graph import evidence_confidence
from app.db import get_session_factory
from app.knowledge.retrieval import retrieve
from evals.ch04.run_eval import load_cases

OUTPUT = Path(__file__).parent / "gate_calibration.json"


def best_threshold(samples: list[dict]) -> dict:
    candidates = [round(i / 100, 2) for i in range(5, 96, 5)]
    answerable_count = sum(sample["answerable"] for sample in samples)
    absent_count = len(samples) - answerable_count
    if not answerable_count or not absent_count:
        raise ValueError("闸值校准样本必须同时包含可回答和库外问题")
    ranked = []
    for threshold in candidates:
        answerable_recall = sum(
            sample["answerable"] and sample["confidence"] >= threshold
            for sample in samples
        ) / answerable_count
        correct_refusal = sum(
            not sample["answerable"] and sample["confidence"] < threshold
            for sample in samples
        ) / absent_count
        f1 = (
            2 * answerable_recall * correct_refusal / (answerable_recall + correct_refusal)
            if answerable_recall + correct_refusal else 0
        )
        correct = sum(
            (sample["confidence"] >= threshold) == sample["answerable"]
            for sample in samples
        )
        ranked.append({
            "threshold": threshold,
            "f1": f1,
            "answerable_recall": answerable_recall,
            "correct_refusal": correct_refusal,
            "accuracy": correct / len(samples),
        })
    return max(ranked, key=lambda item: (item["f1"], item["threshold"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    factory = get_session_factory()
    samples = []
    for index, case in enumerate(cases, 1):
        hits = retrieve(factory, case["query"], top_k=10, strategy="hybrid_rerank")
        signals = evidence_confidence(hits)
        samples.append({
            "id": case["id"],
            "answerable": case["bucket"] != "D_absent",
            **signals,
        })
        print(f"[{index}/{len(cases)}] {case['id']} confidence={signals['confidence']:.4f}")
    result = {"sample_count": len(samples), "recommended": best_threshold(samples), "samples": samples}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["recommended"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
