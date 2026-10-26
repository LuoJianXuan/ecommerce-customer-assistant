"""ch10 多标签贪心分层抽样(纯函数,T5)。

目标:80/10/10 三集,每个标签按比例进三集;计数<3 的稀有标签整类并入训练集;
同种子确定性可复现(语料已冻结,分层清单随 manifest 落盘)。
"""

import random
import unicodedata
from collections import Counter
from typing import Sequence

RARE_LABEL_MIN = 3  # 标签总计数低于此值 → 全部进训练集


def normalized_text_key(text: str) -> str:
    """用于数据去重的稳定键:兼容全半角,忽略空白、标点与大小写。

    训练/验证/测试之间若只差一个问号，本质上仍是同一个样本；用该键做冻结前去重
    和分层后防泄漏校验。保留所有字母与数字，避免把实际内容不同的句子误合并。
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalized if ch.isalnum())


def deduplicate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """按 :func:`normalized_text_key` 去重，返回(保留行,删除数)。

    同一归一化文本若标签冲突则拒绝静默选择，必须先人工裁定。人工复核过的行优先
    于普通行保留，以免丢失 ``review_overridden`` 的溯源信息。
    """
    kept: dict[str, dict] = {}
    for row in rows:
        key = normalized_text_key(row["text"])
        if not key:
            raise ValueError("语料文本归一化后为空")
        previous = kept.get(key)
        if previous is None:
            kept[key] = row
            continue
        if set(previous["labels"]) != set(row["labels"]):
            raise ValueError(
                f"归一化重复文本标签冲突:{previous['text']!r}={previous['labels']} / "
                f"{row['text']!r}={row['labels']}"
            )
        if row.get("review_overridden") and not previous.get("review_overridden"):
            kept[key] = row
    return list(kept.values()), len(rows) - len(kept)


def normalized_overlap(left: list[dict], right: list[dict]) -> set[str]:
    """返回两组语料之间的归一化文本交集。"""
    return {normalized_text_key(r["text"]) for r in left} & {
        normalized_text_key(r["text"]) for r in right
    }


def stratified_split(
    rows: list[dict], ratios: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """贪心配额分层:每行分到「其标签剩余配额比例最高」的集合,集合有容量上限。"""
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(rows)
    caps = [max(1, round(r * n)) for r in ratios]

    label_counts = Counter(l for r in rows for l in r["labels"])
    rare = {l for l, c in label_counts.items() if c < RARE_LABEL_MIN}
    quotas = [{l: c * r for l, c in label_counts.items()} for r in ratios]

    rng = random.Random(seed)
    order = sorted(rows, key=lambda r: (rng.random(), r["text"]))

    sets: list[list[dict]] = [[], [], []]
    used: list[Counter] = [Counter(), Counter(), Counter()]

    for row in order:
        labels = row["labels"]
        if any(l in rare for l in labels):
            chosen = 0
        else:
            best, best_score = 0, float("-inf")
            for s in range(3):
                if len(sets[s]) >= caps[s]:
                    score = float("-inf")
                else:
                    score = sum(
                        (quotas[s][l] - used[s][l]) / max(quotas[s][l], 1e-9) for l in labels
                    )
                if score > best_score:
                    best, best_score = s, score
            chosen = best
        sets[chosen].append(row)
        used[chosen].update(labels)

    return sets[0], sets[1], sets[2]
