"""ch10 T6:RoBERTa-wwm-ext 全参微调多标签主题分类器。

API 依据 Context7 核查的 transformers 5.2 文档(v5 参数名):eval_strategy /
processing_class;多标签:num_labels=17 + problem_type="multi_label_classification"
(BCEWithLogitsLoss 内建);动态 padding(DataCollatorWithPadding)+ max_len=256。
早停盯验证集 macro-F1;weight_decay+warmup 防过拟合;fp16(RTX 2070 支持)。

模型权重经 hf-mirror 下载(HF_ENDPOINT 由 app.config 预置);注意必须先放开
HF_HUB_OFFLINE(app.config 默认置 1 为本地 BGE 离线服务,训练首次需联网取权重)。

用法(项目根目录):
    python training/train.py                     # 常规训练
    python training/train.py --epochs 10 --batch 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # 首次训练需联网下载权重,先于 app.config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config  # noqa: F401,E402  # 预置 HF_ENDPOINT=hf-mirror 等环境变量
from app.topics.taxonomy import TOPIC_NAMES  # noqa: E402
from app.topics.splitting import normalized_overlap  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "model_out"
MODEL_ID = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 256
PATCH = DATA / "targeted_patch.jsonl"


def configure_lr_scheduler(training_args, epochs: int):
    """配置 v5 调度器且保留用户指定轮数。"""
    configured = training_args.set_lr_scheduler(
        name="linear", num_epochs=epochs, warmup_steps=0.1
    )
    assert configured.num_train_epochs == epochs
    return configured


def load_rows(name: str) -> list[dict]:
    return [
        json.loads(l)
        for l in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def load_training_rows(use_patch: bool = True, patch_repeat: int = 1) -> tuple[list[dict], int]:
    """加载基础训练集及仅训练用的定向补丁，并对留出集做防泄漏校验。"""
    rows = load_rows("train")
    if not use_patch or not PATCH.exists():
        return rows, 0
    if patch_repeat < 1:
        raise ValueError("patch_repeat 必须至少为 1")
    patch_rows = [
        json.loads(line)
        for line in PATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdout = load_rows("dev") + load_rows("test")
    overlap = normalized_overlap(patch_rows, holdout)
    if overlap:
        raise ValueError(f"定向补丁与 dev/test 重复,拒绝训练:{len(overlap)} 条")
    internal = normalized_overlap(rows, patch_rows)
    if internal:
        raise ValueError(f"定向补丁与基础训练集重复:{len(internal)} 条")
    effective_patch = patch_rows * patch_repeat
    return rows + effective_patch, len(effective_patch)


def build_dataset(tokenizer, rows: list[dict]):
    import torch
    from app.topics.taxonomy import labels_to_vector

    class TopicDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.enc = tokenizer(
                [r["text"] for r in rows],
                truncation=True,
                max_length=MAX_LEN,
            )
            # 多标签惯例:标签存 float 列表,由 DataCollatorWithPadding 统一成批张量
            # (per-sample 1-D tensor 会触发 features 嵌套错误)
            self.labels = [labels_to_vector(r["labels"]) for r in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = [float(x) for x in self.labels[i]]
            return item

    return TopicDataset(rows)


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    logits, labels = eval_pred
    # 0.5 仅用于各轮 checkpoint 的一致比较；最终推理阈值由 evaluate.py 只在 dev 上选择。
    preds = (1 / (1 + np.exp(-logits))) > 0.5
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    return {"macro_f1": float(f1_macro), "micro_f1": float(f1_micro)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--model", type=str, default=MODEL_ID, help="基础模型或已有检查点")
    ap.add_argument("--no-patch", action="store_true", help="不加载 targeted_patch.jsonl")
    ap.add_argument(
        "--patch-repeat",
        type=int,
        default=1,
        help="定向补丁在训练内的重复采样次数(不改变源文件)",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default="current-run",
        help="检查点子目录名,避免覆盖历史训练检查点",
    )
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    assert torch.cuda.is_available(), "CUDA 不可用:请先安装 CUDA 版 torch(见 docs/training.md)"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    id2label = dict(enumerate(TOPIC_NAMES))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(TOPIC_NAMES),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    train_rows, patch_rows = load_training_rows(
        use_patch=not args.no_patch, patch_repeat=args.patch_repeat
    )
    train_ds = build_dataset(tokenizer, train_rows)
    dev_ds = build_dataset(tokenizer, load_rows("dev"))
    print(
        f"train {len(train_ds)}(定向补丁 {patch_rows}) / dev {len(dev_ds)};"
        f"device={torch.cuda.get_device_name(0)}"
    )

    targs = TrainingArguments(
        output_dir=str(OUT / args.run_name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=42,
    )
    # v5 变更:调度器参数不再走构造函数(warmup_ratio 已废),走 setter;<1 的
    # warmup_steps 按比例解释。注意 setter 的 num_epochs 默认值是 3,若不显式传入会
    # 覆盖上面配置的训练轮数，导致“8 轮+早停”实际固定只跑 3 轮。
    targs = configure_lr_scheduler(targs, args.epochs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    best_dir = OUT / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    metrics = trainer.evaluate()
    summary = {
        "best_dir": str(best_dir),
        "dev_metrics": metrics,
        "args": vars(args),
        "max_len": MAX_LEN,
        "train_rows": len(train_ds),
        "targeted_patch_rows": patch_rows,
        "dev_rows": len(dev_ds),
    }
    (OUT / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"训练完成:dev macro_f1={metrics.get('eval_macro_f1'):.4f} → {best_dir}")


if __name__ == "__main__":
    main()
