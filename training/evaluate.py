"""ch10 T7:测试集逐类目 P/R/F1 + 每类混淆矩阵 + 真实回归集评估 + 错例导出。

产物:evals/ch10/report.md(人读)、evals/ch10/report.json(程序读)、
      evals/ch10/errors.jsonl(测试集错例,人工抽判复核用)。

用法(项目根目录):
    python training/evaluate.py                # 用 training/model_out/best
    python training/evaluate.py --model <目录>
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # 评估不改权重,但 tokenizer 可能需联网校验

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config  # noqa: F401,E402  # 预置 HF_ENDPOINT 等
from app.topics.taxonomy import TOPIC_NAMES, labels_to_vector  # noqa: E402

DATA = Path(__file__).parent / "data"
EVALS = Path(__file__).parent.parent / "evals" / "ch10"
DEFAULT_MODEL = Path(__file__).parent / "model_out" / "best"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def predict(model, tokenizer, texts: list[str], device, batch: int = 64):
    import numpy as np
    import torch

    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(
                texts[i : i + batch],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs)


def metric_summary(y_true, y_pred) -> dict:
    """计算全标签指标及仅对当前集合有金标支持的宏平均。"""
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    active = np.where(y_true.sum(axis=0) > 0)[0]
    supported_macro = precision_recall_fscore_support(
        y_true[:, active], y_pred[:, active], average="macro", zero_division=0
    )[2]
    return {
        "n": len(y_true),
        "macro_f1": round(float(f1_macro), 4),
        "supported_macro_f1": round(float(supported_macro), 4),
        "micro_f1": round(float(f1_micro), 4),
        "exact_match": round(float((y_true == y_pred).all(axis=1).mean()), 4),
        "empty_predictions": int((y_pred.sum(axis=1) == 0).sum()),
    }


def select_threshold(y_true, probs, candidates=None) -> tuple[float, list[dict]]:
    """只用开发集选择全局阈值；先最大化 macro-F1，再比较完全匹配率。

    候选网格固定，避免查看测试集或真实回归集后手工挑阈值造成评估泄漏。
    """
    if candidates is None:
        candidates = [x / 100 for x in range(25, 61, 5)]
    scored = []
    for threshold in candidates:
        metrics = metric_summary(y_true, (probs > threshold).astype(int))
        scored.append({"threshold": threshold, **metrics})
    best = max(
        scored,
        key=lambda x: (x["macro_f1"], x["exact_match"], -abs(x["threshold"] - 0.5)),
    )
    return float(best["threshold"]), scored


def evaluate_set(model, tokenizer, rows, device, threshold: float):
    """返回 (总体指标, 逐类指标, 每类混淆, 行级预测, 错例行)。"""
    import numpy as np
    from sklearn.metrics import (
        multilabel_confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.array([labels_to_vector(r["labels"]) for r in rows])
    probs = predict(model, tokenizer, [r["text"] for r in rows], device)
    y_pred = (probs > threshold).astype(int)

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(TOPIC_NAMES)), zero_division=0
    )
    per_class = {
        name: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(sup[i]),
        }
        for i, name in enumerate(TOPIC_NAMES)
    }
    metrics = metric_summary(y_true, y_pred)
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=range(len(TOPIC_NAMES)))
    conf = {
        name: {"tn": int(m[0][0]), "fp": int(m[0][1]), "fn": int(m[1][0]), "tp": int(m[1][1])}
        for name, m in zip(TOPIC_NAMES, mcm)
    }
    row_preds, errors = [], []
    for row, pred_vec, prob_vec in zip(rows, y_pred, probs):
        pred_labels = [TOPIC_NAMES[i] for i, v in enumerate(pred_vec) if v]
        row_preds.append(pred_labels)
        if pred_labels != row["labels"]:
            errors.append(
                {
                    "text": row["text"],
                    "gold": row["labels"],
                    "pred": pred_labels,
                    "probs": {
                        TOPIC_NAMES[i]: round(float(prob_vec[i]), 3)
                        for i in range(len(TOPIC_NAMES))
                        if prob_vec[i] > 0.2
                    },
                }
            )
    return metrics, per_class, conf, row_preds, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="显式阈值；默认只在 dev 集的固定网格上自动选择",
    )
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    test_rows = load_rows(DATA / "test.jsonl")
    dev_rows = load_rows(DATA / "dev.jsonl")
    real_rows = load_rows(DATA / "real_labeled.jsonl")
    real_eval_rows = [{"text": r["question"], "labels": r["labels"]} for r in real_rows]

    dev_true = np.array([labels_to_vector(r["labels"]) for r in dev_rows])
    dev_probs = predict(model, tokenizer, [r["text"] for r in dev_rows], device)
    if args.threshold is None:
        threshold, threshold_sweep = select_threshold(dev_true, dev_probs)
        threshold_source = "dev_grid_search"
    else:
        if not 0 < args.threshold < 1:
            raise SystemExit("--threshold 必须在 0 和 1 之间")
        threshold = args.threshold
        threshold_sweep = []
        threshold_source = "explicit"

    dev_metrics = metric_summary(dev_true, (dev_probs > threshold).astype(int))
    t_metrics, t_per_class, t_conf, _, t_errors = evaluate_set(
        model, tokenizer, test_rows, device, threshold
    )
    r_metrics, r_per_class, r_conf, r_preds, r_errors = evaluate_set(
        model, tokenizer, real_eval_rows, device, threshold
    )
    challenge_path = DATA / "targeted_challenge.jsonl"
    challenge_rows = load_rows(challenge_path) if challenge_path.exists() else []
    if challenge_rows:
        c_metrics, c_per_class, c_conf, c_preds, c_errors = evaluate_set(
            model, tokenizer, challenge_rows, device, threshold
        )
        pattern_stats = {}
        for row, pred in zip(challenge_rows, c_preds):
            stat = pattern_stats.setdefault(row.get("pattern", "未分组"), {"n": 0, "correct": 0})
            stat["n"] += 1
            stat["correct"] += int(pred == row["labels"])
    else:
        c_metrics = c_per_class = c_conf = pattern_stats = None
        c_errors = []

    EVALS.mkdir(parents=True, exist_ok=True)
    (EVALS / "errors.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in t_errors), encoding="utf-8"
    )
    (EVALS / "challenge_errors.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in c_errors), encoding="utf-8"
    )

    report = {
        "model": args.model,
        "device": device,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_sweep": threshold_sweep,
        "dev": dev_metrics,
        "test": {**t_metrics, "per_class": t_per_class, "confusion": t_conf},
        "real_regression": {**r_metrics, "per_class": r_per_class, "confusion": r_conf},
        "targeted_challenge": (
            {**c_metrics, "per_class": c_per_class, "confusion": c_conf, "patterns": pattern_stats}
            if c_metrics
            else None
        ),
        "errors_exported": len(t_errors),
    }
    (EVALS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model_path = Path(args.model)
    summary_path = model_path.parent / "train_summary.json"
    model_version = model_path.name
    if summary_path.exists():
        try:
            train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            model_version = str(
                train_summary.get("args", {}).get("run_name") or model_version
            )
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    threshold_record = {
        "threshold": threshold,
        "selected_on": "dev" if args.threshold is None else "explicit",
        "dev_metrics": dev_metrics,
        "model_version": model_version,
    }
    if model_path.is_dir():
        (model_path / "topic_threshold.json").write_text(
            json.dumps(threshold_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    lines = [
        "# ch10 主题分类器评估报告",
        "",
        f"模型:`{args.model}`  设备:{device}  阈值:{threshold:.2f}({threshold_source})",
        "",
        f"开发集选阈值指标:macro-F1={dev_metrics['macro_f1']},完全匹配率={dev_metrics['exact_match']}",
        "",
        "## 测试集(分层留出,非增强)",
        "",
        f"n={t_metrics['n']}  **macro-F1={t_metrics['macro_f1']}**  micro-F1={t_metrics['micro_f1']}  完全匹配率={t_metrics['exact_match']}",
        "",
        "| 类目 | P | R | F1 | support | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in TOPIC_NAMES:
        pc, cf = t_per_class[name], t_conf[name]
        lines.append(
            f"| {name} | {pc['precision']:.2f} | {pc['recall']:.2f} | **{pc['f1']:.2f}** | {pc['support']} | {cf['tp']} | {cf['fp']} | {cf['fn']} |"
        )
    if c_metrics:
        lines += [
            "",
            "## 定向 challenge 集(全新人工金标,未参与训练与阈值选择)",
            "",
            f"n={c_metrics['n']}  **macro-F1={c_metrics['macro_f1']}**  "
            f"有金标支持类 macro-F1={c_metrics['supported_macro_f1']}  "
            f"micro-F1={c_metrics['micro_f1']}  完全匹配率={c_metrics['exact_match']}",
            "",
            "| 模式 | 正确/总数 |",
            "|---|---|",
        ]
        for pattern, stat in pattern_stats.items():
            lines.append(f"| {pattern} | {stat['correct']}/{stat['n']} |")
    lines += [
        "",
        "## 真实回归集(26 条低置信度问题去重后 12 条,未参与训练)",
        "",
        f"n={r_metrics['n']}  **全 17 类 macro-F1={r_metrics['macro_f1']}**  "
        f"有金标支持类 macro-F1={r_metrics['supported_macro_f1']}  "
        f"micro-F1={r_metrics['micro_f1']}  完全匹配率={r_metrics['exact_match']}",
        "",
        "| 真实问题 | 金标 | 预测 |",
        "|---|---|---|",
    ]
    for row, pred in zip(real_rows, r_preds):
        q = row["question"].replace("|", "\\|")[:50] + ("…" if len(row["question"]) > 50 else "")
        flag = "✅" if pred == row["labels"] else "❌"
        lines.append(f"| {q} | {','.join(row['labels'])} | {flag}{','.join(pred)} |")
    lines += ["", f"测试集错例 {len(t_errors)} 条已导出 `evals/ch10/errors.jsonl`(人工抽判复核用)。"]
    (EVALS / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"dev 选择阈值={threshold:.2f};测试集 macro-F1={t_metrics['macro_f1']} "
        f"micro-F1={t_metrics['micro_f1']} 错例 {len(t_errors)} 条"
    )
    if c_metrics:
        print(
            f"定向 challenge 完全匹配={c_metrics['exact_match']} 错例 {len(c_errors)} 条"
        )
    print(f"真实回归集 macro-F1={r_metrics['macro_f1']} 完全匹配 {r_metrics['exact_match']} 错例 {len(r_errors)} 条")
    print(f"报告 → {EVALS / 'report.md'};错例 → {EVALS / 'errors.jsonl'}")


if __name__ == "__main__":
    main()
