"""ch10 T5:冻结语料分层 80/10/10 + 训练集增强(同义词 + LLM 句式微调)。

输入 training/data/synthetic.jsonl(须与 corpus_manifest.json 的 sha256 一致,防冻结后漂移);
输出 train.jsonl / dev.jsonl / test.jsonl + split_manifest.json(三集计数、标签分布、哈希)。
增强只作用于训练集:同义词替换约 20% 行,LLM 句式微调约 400 句(标签原样复制)。

用法(项目根目录):
    python training/split_augment.py
    python training/split_augment.py --no-llm   # 跳过 LLM 句式微调(调试)
"""

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import build_chat_model  # noqa: E402
from app.topics.augment import synonym_augment  # noqa: E402
from app.topics.splitting import normalized_overlap, normalized_text_key, stratified_split  # noqa: E402

DATA = Path(__file__).parent / "data"
SYNTHETIC = DATA / "synthetic.jsonl"
MANIFEST = DATA / "corpus_manifest.json"

SEED = 42
SYN_RATE = 0.2
PARA_TARGET = 400
PARA_BATCH = 20

PARAPHRASE_PROMPT = """你是电商客服语料增强助手。把下面带编号的用户问题逐条**换一种口语说法**:
保持意思、诉求数量和语气完全一致,长度变化不超过 ±5 字,仍然像用户打字;
只换句式和用词,不要书面腔。严格返回 JSON 数组,元素形如 {{"id": <编号>, "text": "<改写>"}},不要输出其他文字:

{payload}"""


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def paraphrase(rows: list[dict]) -> list[dict]:
    """LLM 句式微调(标签复制,仅训练集用)。"""
    llm = build_chat_model(get_settings()).bind(temperature=0.8)
    sem = asyncio.Semaphore(4)
    out: list[dict] = []

    async def one(batch: list[dict]) -> None:
        payload = json.dumps(
            [{"id": i, "q": r["text"]} for i, r in enumerate(batch)], ensure_ascii=False
        )
        async with sem:
            resp = await asyncio.wait_for(
                llm.ainvoke(PARAPHRASE_PROMPT.format(payload=payload)), timeout=90
            )
        for item in _parse_json_array(resp.content):
            i, text = item.get("id"), item.get("text")
            if isinstance(i, int) and 0 <= i < len(batch) and isinstance(text, str) and 4 <= len(text) <= 60:
                src = batch[i]
                out.append(
                    {"text": text.strip(), "labels": src["labels"], "primary": src["primary"], "augmented": "paraphrase"}
                )

    batches = [rows[i : i + PARA_BATCH] for i in range(0, len(rows), PARA_BATCH)]
    await asyncio.gather(*(one(b) for b in batches))
    return out[:PARA_TARGET]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    # 冻结完整性校验
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = _sha256(SYNTHETIC)
    if actual != manifest["synthetic_sha256"]:
        raise SystemExit("synthetic.jsonl 与冻结清单哈希不一致,拒绝分层(先核对语料是否被改动)")

    rows = [json.loads(l) for l in SYNTHETIC.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, dev, test = stratified_split(rows, ratios=(0.8, 0.1, 0.1), seed=SEED)
    base_overlaps = {
        "train_dev": len(normalized_overlap(train, dev)),
        "train_test": len(normalized_overlap(train, test)),
        "dev_test": len(normalized_overlap(dev, test)),
    }
    if any(base_overlaps.values()):
        raise SystemExit(f"冻结语料存在归一化重复,拒绝产生泄漏分层:{base_overlaps}")
    n_train0 = len(train)
    holdout_keys = {normalized_text_key(r["text"]) for r in dev + test}
    train_keys = {normalized_text_key(r["text"]) for r in train}

    # 同义词增强(确定性,seed 派生自行号)
    import random

    rng = random.Random(SEED)
    idx = rng.sample(range(len(train)), k=int(len(train) * SYN_RATE))
    syn_added = 0
    for i in idx:
        variant = synonym_augment(train[i]["text"], seed=SEED * 1000 + i)
        key = normalized_text_key(variant) if variant else ""
        if variant and key not in train_keys and key not in holdout_keys:
            train.append(
                {"text": variant, "labels": train[i]["labels"], "primary": train[i]["primary"], "augmented": "synonym"}
            )
            train_keys.add(key)
            syn_added += 1

    # LLM 句式微调
    para_added = 0
    if not args.no_llm:
        base_rows = [r for r in train if not r.get("augmented")]
        picked = random.Random(SEED + 1).sample(base_rows, k=min(PARA_TARGET, len(base_rows)))
        para = asyncio.run(paraphrase(picked))
        # 去重后追加
        for r in para:
            key = normalized_text_key(r["text"])
            if key not in train_keys and key not in holdout_keys:
                train.append(r)
                train_keys.add(key)
                para_added += 1

    final_overlaps = {
        "train_dev": len(normalized_overlap(train, dev)),
        "train_test": len(normalized_overlap(train, test)),
        "dev_test": len(normalized_overlap(dev, test)),
    }
    if any(final_overlaps.values()):
        raise RuntimeError(f"增强后出现跨分片归一化重复:{final_overlaps}")

    for name, data in (("train", train), ("dev", dev), ("test", test)):
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def label_dist(rs: list[dict]) -> dict[str, int]:
        return dict(sorted(Counter(l for r in rs for l in r["labels"]).items()))

    split_manifest = {
        "seed": SEED,
        "corpus_sha256": actual,
        "rows": {"train": len(train), "dev": len(dev), "test": len(test)},
        "train_before_augment": n_train0,
        "augmented": {"synonym": syn_added, "paraphrase": para_added},
        "normalized_overlap": final_overlaps,
        "label_distribution": {k: label_dist(v) for k, v in (("train", train), ("dev", dev), ("test", test))},
        "files_sha256": {f"{n}.jsonl": _sha256(DATA / f"{n}.jsonl") for n in ("train", "dev", "test")},
    }
    (DATA / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"分层:train {n_train0} + 增强(同义 {syn_added}/句式 {para_added}) = {len(train)},"
        f"dev {len(dev)},test {len(test)};清单 → split_manifest.json"
    )


if __name__ == "__main__":
    main()
