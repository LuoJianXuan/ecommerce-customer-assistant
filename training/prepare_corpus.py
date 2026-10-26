"""ch10 数据第一步:真实问题清洗流水线(清洗→脱敏→重复压缩→去重→LLM 修错别字)。

输入 training/data/real_questions.jsonl(dump 提取),输出 training/data/real_clean.jsonl。
真实问题只作"真实回归评估集",不进训练集(计划已确认决策)。

用法(项目根目录):
    python training/prepare_corpus.py            # 全流程(含 LLM 修错别字)
    python training/prepare_corpus.py --no-llm   # 跳过 LLM,仅纯函数清洗
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.topics.cleaning import (  # noqa: E402
    collapse_repeated_text,
    dedupe_questions,
    desensitize,
    normalize_text,
)

DATA = Path(__file__).parent / "data"
SRC = DATA / "real_questions.jsonl"
OUT = DATA / "real_clean.jsonl"

TYPO_FIX_PROMPT = """你是客服语料清洗助手。下面是带编号的电商客服用户问题列表(JSON)。
请只修正其中的**错别字/明显笔误**(如"我得运费"→"我的运费"、"优惠卷"→"优惠券"),
保持口语风格、语序和原意完全不变;没有错字的条目原样返回;不要增删内容、不要书面化改写。
严格返回 JSON 数组,每个元素形如 {{"id": <原编号>, "fixed": "<修正后问题>"}},不要输出其他文字。

{payload}"""


def load_rows() -> list[dict]:
    return [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]


def pure_clean(rows: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """归一化→脱敏→重复段压缩→按问题去重,返回(唯一行, 出现次数)。"""
    prepared = [
        {
            "id": r["id"],
            "question": collapse_repeated_text(desensitize(normalize_text(r["raw_question"]))),
            "source": r["source"],
        }
        for r in rows
    ]
    return dedupe_questions(prepared)


def llm_fix_typos(rows: list[dict]) -> list[dict]:
    """LLM 只修错别字;失败/解析异常时原样返回(宁可不动,不可改写)。"""
    import asyncio

    from app.llm import build_chat_model

    async def run() -> list[dict]:
        payload = json.dumps(
            [{"id": r["id"], "question": r["question"]} for r in rows], ensure_ascii=False
        )
        llm = build_chat_model(_get_settings())  # type: ignore[name-defined]
        resp = await llm.ainvoke(TYPO_FIX_PROMPT.format(payload=payload))
        return json.loads(resp.content)

    try:
        fixed = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 —— 清洗失败不阻塞,回退原文本
        print(f"[warn] LLM 修错别字失败,保留原文: {e}")
        return rows
    by_id = {r["id"]: r for r in rows}
    for item in fixed:
        rid, text = item.get("id"), item.get("fixed")
        if rid in by_id and isinstance(text, str) and text.strip():
            by_id[rid]["question"] = text.strip()
    return rows


def _get_settings():
    from app.config import get_settings

    return get_settings()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 修错别字")
    args = ap.parse_args()

    rows = load_rows()
    unique, counts = pure_clean(rows)
    print(f"原始 {len(rows)} 条 → 去重后 {len(unique)} 个唯一问题(重复计数已记录)")

    if not args.no_llm:
        unique = llm_fix_typos(unique)

    with OUT.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(
                json.dumps(
                    {
                        "orig_id": r["id"],
                        "question": r["question"],
                        "source": r["source"],
                        "dup_count": counts[r["id"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"清洗完成 → {OUT}")
    for r in unique:
        preview = r["question"][:36] + ("…" if len(r["question"]) > 36 else "")
        print(f"  [{r['id']}] (×{counts[r['id']]}) {preview}")


if __name__ == "__main__":
    main()
