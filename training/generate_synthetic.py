"""ch10 数据第二步:LLM 照术语表造数 + 复判自检(含定向补数模式)。

全量模式:每类目标 per_class(默认 300)句,约 1/4 带第二独立诉求(多标签);
全量 NFKC 去重(含与真实问题);温度 0 独立复判轮校验标签一致性,不一致丢弃并记录分歧。
补数模式(--topup):读现有 synthetic.jsonl 各主类缺口,只补不足的类,补到目标(复判淘汰后继续补)。
产出 training/data/synthetic.jsonl;分歧记录 training/data/verify_disagreements.jsonl(追加)。

用法(项目根目录):
    python training/generate_synthetic.py                     # 全量重建
    python training/generate_synthetic.py --per-class 8       # 冒烟
    python training/generate_synthetic.py --topup             # 定向补数到每类 per_class
    python training/generate_synthetic.py --topup --skip-verify   # 调试
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import build_chat_model  # noqa: E402
from app.topics.cleaning import normalize_text  # noqa: E402
from app.topics.splitting import normalized_text_key  # noqa: E402
from app.topics.taxonomy import TOPICS, taxonomy_prompt_block, validate_labels  # noqa: E402

DATA = Path(__file__).parent / "data"
REAL = DATA / "real_clean.jsonl"
OUT = DATA / "synthetic.jsonl"
DISAGREE_LOG = DATA / "verify_disagreements.jsonl"

BATCH = 30          # 单次生成条数
VERIFY_BATCH = 20   # 单次复判条数
CONCURRENCY = 5

GEN_PROMPT = """你是电商客服语料造数助手。
{taxonomy}

现在为类目「{topic}」生成 {n} 条**用户问题**语料,要求:
1. 口语化、真实电商客服口吻,像真用户打字(可以带语气词,禁止书面腔和客服腔);
2. 长度 5~40 字,句式多样(疑问/陈述/省略/带情绪),禁止模板化重复开头;
3. 其中约四分之一的句子自然带上第二个**需要客服独立回答或处理**的诉求并如实标多标签(如「买大了想退」→ ["退换货","尺码"]);背景信息(订单经过、支付经历、商品来源、身份限定词)不单独打标签;其余只标 ["{topic}"];
4. 不造含手机号/订单号/具体金额数字的句子;
5. 严格返回 JSON 数组,元素形如 {{"text": "...", "labels": [...]}},不要输出任何其他文字。只输出数组本身。"""

VERIFY_PROMPT = """你是电商客服问题主题标注员。
{taxonomy}

给下面带编号的问题逐条打多标签(按「独立诉求」口径,见上文示例;无法归类打["其他"])。
严格返回 JSON 数组,元素形如 {{"id": <编号>, "labels": [...]}},不要输出任何其他文字:

{payload}"""


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


async def _call_llm(llm, prompt: str, sem: asyncio.Semaphore, retries: int = 1) -> str:
    for attempt in range(retries + 1):
        try:
            async with sem:
                resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=90)
            return resp.content
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
    raise RuntimeError("unreachable")


def _valid_item(item: dict, topic_name: str | None) -> dict | None:
    text, labels = item.get("text"), item.get("labels") or []
    if not isinstance(text, str):
        return None
    text = normalize_text(text)
    if not text or len(text) < 4 or len(text) > 60:
        return None
    try:
        valid = validate_labels(labels)
    except ValueError:
        return None
    if topic_name and topic_name not in valid:  # 主类必须是造数目标类
        return None
    return {"text": text, "labels": valid}


async def generate_for_topic(
    llm, topic_name: str, want: int, sem, global_seen: set[str]
) -> list[dict]:
    """生成 want 句(主类=topic_name),全量去重后即返回;want 小时单批即可。"""
    collected: list[dict] = []
    for _ in range(max(1, -(-want // BATCH) + 3)):  # 批次余量,凑够即停
        raw = await _call_llm(
            llm, GEN_PROMPT.format(taxonomy=taxonomy_prompt_block(), topic=topic_name, n=BATCH), sem
        )
        for item in _parse_json_array(raw):
            rec = _valid_item(item, topic_name)
            key = normalized_text_key(rec["text"]) if rec else ""
            if rec and key not in global_seen:
                global_seen.add(key)
                rec["primary"] = topic_name
                collected.append(rec)
        if len(collected) >= want:
            break
    return collected[:want]


async def verify_labels(llm, rows: list[dict], sem) -> tuple[list[dict], list[dict]]:
    """温度 0 独立复判:预测标签集≠声明的丢弃并记录分歧;返回(保留, 分歧记录)。"""
    llm0 = build_chat_model(get_settings()).bind(temperature=0)
    kept: list[dict] = []
    disagreements: list[dict] = []

    async def one(batch: list[dict]) -> None:
        payload = json.dumps(
            [{"id": i, "q": r["text"]} for i, r in enumerate(batch)], ensure_ascii=False
        )
        raw = await _call_llm(
            llm0, VERIFY_PROMPT.format(taxonomy=taxonomy_prompt_block(), payload=payload), sem
        )
        preds = {item["id"]: set(item.get("labels") or []) for item in _parse_json_array(raw)}
        for i, r in enumerate(batch):
            if preds.get(i) == set(r["labels"]):
                kept.append(r)
            else:
                disagreements.append(
                    {"text": r["text"], "declared": r["labels"], "predicted": sorted(preds.get(i, []))}
                )

    batches = [rows[i : i + VERIFY_BATCH] for i in range(0, len(rows), VERIFY_BATCH)]
    await asyncio.gather(*(one(b) for b in batches))
    return kept, disagreements


def _append_disagreements(disagreements: list[dict]) -> None:
    if not disagreements:
        return
    with DISAGREE_LOG.open("a", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _summary(all_rows: list[dict]) -> str:
    by_primary: dict[str, int] = {t.name: 0 for t in TOPICS}
    for r in all_rows:
        by_primary[r["primary"]] = by_primary.get(r["primary"], 0) + 1
    multi = sum(1 for r in all_rows if len(r["labels"]) > 1)
    avg_len = sum(len(r["text"]) for r in all_rows) / max(1, len(all_rows))
    dist = " ".join(f"{k}:{v}" for k, v in by_primary.items())
    return (
        f"共 {len(all_rows)} 句(主类分布 {dist});多标签 {multi} 句"
        f"({multi / max(1, len(all_rows)):.0%});平均 {avg_len:.1f} 字"
    )


async def run_full(args) -> None:
    real_texts = set()
    if REAL.exists():
        real_texts = {
            normalize_text(json.loads(l)["question"])
            for l in REAL.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
    llm = build_chat_model(get_settings()).bind(temperature=0.9)
    sem = asyncio.Semaphore(CONCURRENCY)

    all_rows: list[dict] = []
    global_seen: set[str] = {normalized_text_key(text) for text in real_texts}
    for t in TOPICS:
        rows = await generate_for_topic(llm, t.name, args.per_class, sem, global_seen)
        all_rows.extend(rows)
        print(f"  {t.name}: +{len(rows)}(累计 {len(all_rows)})", flush=True)

    if not args.skip_verify:
        all_rows, disagreements = await verify_labels(llm, all_rows, sem)
        print(f"复判:保留 {len(all_rows)},丢弃 {len(disagreements)}", flush=True)
        _append_disagreements(disagreements)

    with OUT.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"产出 → {OUT}:{_summary(all_rows)}")


async def run_topup(args) -> None:
    """定向补数:只补主类计数 < per_class 的类,复判淘汰后继续补到目标。"""
    if not OUT.exists():
        raise SystemExit("synthetic.jsonl 不存在,请先跑全量模式")
    existing = [
        json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    by_primary: dict[str, list[dict]] = {t.name: [] for t in TOPICS}
    for r in existing:
        by_primary.setdefault(r["primary"], []).append(r)

    llm = build_chat_model(get_settings()).bind(temperature=0.9)
    sem = asyncio.Semaphore(CONCURRENCY)
    global_seen: set[str] = {normalized_text_key(r["text"]) for r in existing}
    if REAL.exists():
        global_seen |= {
            normalized_text_key(normalize_text(json.loads(l)["question"]))
            for l in REAL.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }

    added_total = 0
    for t in TOPICS:
        deficit = args.per_class - len(by_primary[t.name])
        if deficit <= 0:
            continue
        print(f"  {t.name}: 现有 {len(by_primary[t.name])},补 {deficit}", flush=True)
        got = 0
        attempts = 0
        while got < deficit and attempts < 8:
            attempts += 1
            batch = await generate_for_topic(llm, t.name, min(BATCH, deficit - got + 8), sem, global_seen)
            if not batch:
                break
            if not args.skip_verify:
                batch, disagreements = await verify_labels(llm, batch, sem)
                _append_disagreements(disagreements)
            batch = batch[: deficit - got]
            by_primary[t.name].extend(batch)
            got += len(batch)
        added_total += got
        print(f"    实补 +{got}(现 {len(by_primary[t.name])})", flush=True)

    all_rows = [r for t in TOPICS for r in by_primary[t.name]]
    with OUT.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"补数完成,新增 {added_total} → {OUT}:{_summary(all_rows)}")


async def run_reverify(args) -> None:
    """用当前口径全量复判已有语料:不一致的丢弃并记分歧(统一标注标准用)。"""
    if not OUT.exists():
        raise SystemExit("synthetic.jsonl 不存在")
    existing = [
        json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    llm = build_chat_model(get_settings()).bind(temperature=0)
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"重复判 {len(existing)} 句(独立诉求口径)…", flush=True)
    kept, disagreements = await verify_labels(llm, existing, sem)
    _append_disagreements(disagreements)
    with OUT.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"重复判:保留 {len(kept)},丢弃 {len(disagreements)} → {OUT}", flush=True)
    print(_summary(kept), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--topup", action="store_true", help="定向补数模式")
    ap.add_argument("--reverify", action="store_true", help="按当前口径全量重复判已有语料")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    if args.reverify:
        print("重复判模式", flush=True)
        asyncio.run(run_reverify(args))
        return
    print("补数模式" if args.topup else "全量模式", f"每类目标 {args.per_class}", flush=True)
    if args.topup:
        asyncio.run(run_topup(args))
    else:
        asyncio.run(run_full(args))


if __name__ == "__main__":
    main()
