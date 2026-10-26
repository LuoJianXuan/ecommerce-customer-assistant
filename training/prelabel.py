"""ch10 数据第三步:真实问题 LLM 预标 + 人工抽审样本导出。

对 real_clean.jsonl 的唯一真实问题照术语表预标多标签(温度 0);
再从真实+造数语料抽 ~50 条导出成 training/data/review_sample.md 供人工抽审(评审门)。

用法(项目根目录):
    python training/prelabel.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import build_chat_model  # noqa: E402
from app.topics.taxonomy import (  # noqa: E402
    TOPICS,
    taxonomy_prompt_block,
    validate_labels,
)

DATA = Path(__file__).parent / "data"
REAL_CLEAN = DATA / "real_clean.jsonl"
SYNTHETIC = DATA / "synthetic.jsonl"
DISAGREE_LOG = DATA / "verify_disagreements.jsonl"
OUT_LABELED = DATA / "real_labeled.jsonl"
OUT_SAMPLE = DATA / "review_sample.md"

PRELABEL_PROMPT = """你是电商客服问题主题标注员。
{taxonomy}

给下面带编号的真实用户问题逐条打多标签:字面提到几个诉求打几个,一个不多一个不少;无法归类打["其他"]。
严格返回 JSON 数组,元素形如 {{"id": <编号>, "labels": [...]}},不要输出任何其他文字:

{payload}"""


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


async def prelabel() -> list[dict]:
    rows = [
        json.loads(l)
        for l in REAL_CLEAN.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    llm = build_chat_model(get_settings()).bind(temperature=0)
    payload = json.dumps(
        [{"id": i, "q": r["question"]} for i, r in enumerate(rows)], ensure_ascii=False
    )
    resp = await llm.ainvoke(PRELABEL_PROMPT.format(taxonomy=taxonomy_prompt_block(), payload=payload))
    preds = {item["id"]: item.get("labels") or [] for item in _parse_json_array(resp.content)}

    labeled = []
    for i, r in enumerate(rows):
        try:
            labels = validate_labels(preds.get(i, []))
        except ValueError:
            labels = ["其他"]  # 预标兜底,抽审时可纠正
        labeled.append({**r, "labels": labels})
    return labeled


def export_review_sample(labeled: list[dict]) -> None:
    """v2(抽审门修订):全文显示+附 orig_id;分层扩样=真实全量+每类5条+30条特殊样本。"""
    random.seed(10)
    synth = []
    if SYNTHETIC.exists():
        synth = [
            json.loads(l)
            for l in SYNTHETIC.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    disagreements = []
    if DISAGREE_LOG.exists():
        disagreements = [
            json.loads(l)
            for l in DISAGREE_LOG.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ][-30:]

    NEAR_PAIRS = {frozenset(p) for p in (("退换货", "保修维修"), ("物流", "运费"), ("优惠活动", "价保"))}

    def pick_per_class(n: int = 5) -> list[dict]:
        out = []
        for name in [t.name for t in TOPICS]:
            pool = [r for r in synth if r["primary"] == name]
            out.extend(random.sample(pool, min(n, len(pool))))
        return out

    def pick_special(total: int = 30) -> list[dict]:
        multi = [r for r in synth if len(r["labels"]) > 1]
        near = [r for r in synth if frozenset(r["labels"]) in NEAR_PAIRS]
        picks: list[dict] = []
        for bucket, n in ((near, 10), (multi, 20)):
            picks.extend(random.sample(bucket, min(n, len(bucket))))
        return picks[:total]

    lines = [
        "# ch10 标注抽审样本 v2(二次评审门)",
        "",
        "口径(已修订):**每个需要客服独立回答或处理的明确诉求打一个标签**;"
        "背景信息(订单经过、支付经历、商品来源、身份限定词)不单独打标签。",
        "近邻重点:修/退、运费(钱)/物流(货)、价保(补差)/优惠活动(券和满减)。",
        "错的标签在「你的修正」列写正确标签;分歧区给出你认为对的一方。",
        "",
        "## 一、真实问题(全量 %d 条,含 orig_id 与全文)" % len(labeled),
        "",
        "| orig_id | 问题(全文) | 预标标签 | 你的修正 |",
        "|---|---|---|---|",
    ]
    for r in labeled:
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {r['orig_id']} | {q} | {','.join(r['labels'])} | |")

    per_class = pick_per_class(5)
    lines += [
        "",
        "## 二、造数语料分层抽样(每类 5 条,共 %d 条)" % len(per_class),
        "",
        "| # | 句子 | 标签 | 你的修正 |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(per_class, 1):
        lines.append(f"| {i} | {r['text']} | {','.join(r['labels'])} | |")

    special = pick_special(30)
    lines += [
        "",
        "## 三、特殊样本:多标签/近邻边界(共 %d 条)" % len(special),
        "",
        "| # | 句子 | 标签 | 你的修正 |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(special, 1):
        lines.append(f"| {i} | {r['text']} | {','.join(r['labels'])} | |")

    lines += [
        "",
        "## 四、复判分歧样本(被丢弃的边界句,裁定谁对;共 %d 条)" % len(disagreements),
        "",
        "| # | 句子 | 造数声明 | 复判认为 | 你的裁定 |",
        "|---|---|---|---|---|",
    ]
    for i, d in enumerate(disagreements[-20:], 1):
        text = d["text"].replace("|", "\\|")
        lines.append(f"| {i} | {text} | {','.join(d['declared'])} | {','.join(d['predicted'])} | |")

    lines += ["", "---", "审完交回;成批错误请说明类目与原因,我会调 prompt 重跑。"]
    OUT_SAMPLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    labeled = asyncio.run(prelabel())
    with OUT_LABELED.open("w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    export_review_sample(labeled)
    print(f"预标 {len(labeled)} 条真实问题 → {OUT_LABELED}")
    for r in labeled:
        q = r["question"][:30] + ("…" if len(r["question"]) > 30 else "")
        print(f"  [{r['orig_id']}] {','.join(r['labels'])} | {q}")
    print(f"抽审样本已导出 → {OUT_SAMPLE}")


if __name__ == "__main__":
    main()
