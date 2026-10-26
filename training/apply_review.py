"""ch10 T4.5:落实二次抽审结论到权威语料(人工裁定压过一切 LLM 结果)。

三种动作:
1. 机械化修正(成批规则 1):「其他」与可归类标签共存的行,去掉「其他」;
2. 应用 review_sample.md 里的 8 条人工标签修正(按句子文本匹配);
3. 分歧裁定落实:裁定=复判侧或第三方标签的,维持丢弃(不回补);若裁定=造数声明侧,
   把该句按声明标签回补进语料(本轮无此情形,逻辑保留)。
最后可选冻结:生成 corpus_manifest.json(计数+SHA256+种子),作为 T5 分层的唯一输入。

用法(项目根目录):
    python training/apply_review.py            # 应用修正并打印统计
    python training/apply_review.py --freeze   # 应用修正并写入冻结清单
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.topics.splitting import deduplicate_rows, normalized_text_key  # noqa: E402
from app.topics.taxonomy import validate_labels  # noqa: E402

DATA = Path(__file__).parent / "data"
SAMPLE = DATA / "review_sample.md"
SYNTHETIC = DATA / "synthetic.jsonl"
MANIFEST = DATA / "corpus_manifest.json"

SPLIT_SEED = 42


def parse_corrections() -> tuple[list[dict], list[dict]]:
    """解析 review_sample.md 的「你的修正」与「你的裁定」两列。

    返回 (标签修正列表[{text,new_labels}], 分歧裁定列表[{text,declared,predicted,ruled}])。
    """
    corrections: list[dict] = []
    rulings: list[dict] = []
    section = ""
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or set(cells[0]) <= set("#-"):
            continue
        if "你的裁定" in line or section.startswith("## 四"):
            if len(cells) >= 5 and cells[4]:
                rulings.append(
                    {
                        "text": cells[1],
                        "declared": [x for x in cells[2].split(",") if x],
                        "predicted": [x for x in cells[3].split(",") if x],
                        "ruled": [x for x in cells[4].split(",") if x],
                    }
                )
        elif cells[-1] and "你的修正" not in line and "预标标签" not in line:
            # 造数区表:| # | 句子 | 标签 | 你的修正 |;真实区表多一列 orig_id,修正也在末列
            text = cells[-3] if len(cells) >= 4 else ""
            new_labels = [x for x in cells[-1].split(",") if x and x != "→"]
            if text and new_labels:
                corrections.append({"text": text, "new_labels": new_labels})
    return corrections, rulings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="写入 corpus_manifest.json 冻结语料")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SYNTHETIC.read_text(encoding="utf-8").splitlines() if l.strip()]
    corrections, rulings = parse_corrections()

    # 1) 机械修正:「其他」与可归类标签共存 → 去掉「其他」(成批规则 1)
    stripped = 0
    for r in rows:
        r["labels"] = validate_labels(r["labels"])
        if "其他" in r["labels"] and len(r["labels"]) > 1:
            r["labels"] = [x for x in r["labels"] if x != "其他"]
            stripped += 1

    # 2) 应用人工标签修正(权威,压过 LLM):句子在语料中→改标签;
    #    不在(被重复判按新规则丢弃的违规句)→按用户标签回补——用户裁定=金标,
    #    句子本身是优质训练数据,不该因旧标签错误而丢失。
    by_text = {}
    for r in rows:
        by_text.setdefault(r["text"], r)
    applied, reinstated_corr, missing = 0, 0, []
    for c in corrections:
        r = by_text.get(c["text"])
        labels = validate_labels(c["new_labels"])
        if r is None:
            rec = {"text": c["text"], "labels": labels, "primary": labels[0], "review_overridden": True}
            rows.append(rec)
            by_text[c["text"]] = rec
            reinstated_corr += 1
            continue
        r["labels"] = labels
        r["review_overridden"] = True
        applied += 1

    # 3) 分歧裁定:裁定=声明侧 → 回补;否则维持丢弃
    reinstated = 0
    if rulings:
        for d in rulings:
            if sorted(d["ruled"]) == sorted(d["declared"]) and d["text"] not in by_text:
                rec = {
                    "text": d["text"],
                    "labels": validate_labels(d["declared"]),
                    "primary": d["declared"][0],
                    "review_overridden": True,
                }
                rows.append(rec)
                by_text[d["text"]] = rec
                reinstated += 1

    # 4) 冻结前不变量:主类必须仍在人工裁定后的标签中；标点/空白变体只保留一份。
    primary_repaired = 0
    for r in rows:
        if r.get("primary") not in r["labels"]:
            r["primary"] = r["labels"][0]
            primary_repaired += 1
    rows, deduplicated = deduplicate_rows(rows)

    with SYNTHETIC.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"语料 {len(rows)} 行:其他共存剥离 {stripped},修正就地应用 {applied},"
        f"修正回补 {reinstated_corr},分歧回补 {reinstated},"
        f"主类修复 {primary_repaired},归一化去重 {deduplicated}"
    )

    if args.freeze:
        digest = hashlib.sha256(SYNTHETIC.read_bytes()).hexdigest()
        by_primary: dict[str, int] = {}
        for r in rows:
            by_primary[r["primary"]] = by_primary.get(r["primary"], 0) + 1
        manifest = {
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
            "synthetic_rows": len(rows),
            "synthetic_sha256": digest,
            "primary_counts": by_primary,
            "real_labeled_rows": len(
                (DATA / "real_labeled.jsonl").read_text(encoding="utf-8").splitlines()
            ),
            "split_seed": SPLIT_SEED,
            "review_applied": {
                "label_corrections": applied,
                "corrections_reinstated": reinstated_corr,
                "other_stripped": stripped,
                "disagreement_reinstated": reinstated,
                "primary_repaired": primary_repaired,
                "normalized_duplicates_removed": deduplicated,
            },
            "invariants": {
                "primary_in_labels": all(r["primary"] in r["labels"] for r in rows),
                "normalized_text_unique": len(rows)
                == len({normalized_text_key(r["text"]) for r in rows}),
            },
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"冻结清单 → {MANIFEST}(sha256={digest[:16]}…,seed={SPLIT_SEED})")


if __name__ == "__main__":
    main()
