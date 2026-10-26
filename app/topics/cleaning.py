"""ch10 语料清洗纯函数:归一化、脱敏、重复段压缩、按问题去重。

数据事实(T0 提取发现):真实池 26 条去重后仅约 12 个唯一问题,存在同段文本
粘贴 3 遍的 608 字长句与"我得运费/优惠卷"类错别字——前三类问题由本模块的
纯函数解决,错别字由 prepare_corpus.py 调 LLM 修(仅真实问题,量小)。
"""

import re
import unicodedata
from typing import TypedDict

_WS_RE = re.compile(r"\s+")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ORDER_RE = re.compile(r"(?i)(?<![0-9a-zA-Z])so\d{6,}(?![0-9])")
_AMOUNT_RE = re.compile(r"\d+(?:\.\d{1,2})?\s*[元块]")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")


def normalize_text(text: str) -> str:
    """NFKC 归一(全半角统一)+ 空白折叠;与 mine.py 判重口径一致。"""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def desensitize(text: str) -> str:
    """手机号/订单号/金额 → 占位符(顺序先手机后金额,防误吞)。"""
    text = _PHONE_RE.sub("<手机号>", text)
    text = _ORDER_RE.sub("<订单号>", text)
    return _AMOUNT_RE.sub("<金额>", text)


def collapse_repeated_text(text: str, min_sentence_len: int = 15) -> str:
    """压缩同段文本的连续重复粘贴。

    两步:①后缀周期折叠:任意「前缀 + 重复块×k」只留前缀+一份块(含整串自重复
    的无前缀情形)。真实池 608 字长句=23 字开头+195 字块×3,旧算法只处理无前缀
    情形漏折,句级去重也只消掉相邻一块——抽审门发现后重写。块长下限 16 字:
    粘贴错误的块都是长段,「谢谢/好，好」类口语叠字不受影响;
    ②相邻重复长句去重。
    """
    n = len(text)
    min_block = 16
    if n >= 2 * min_block:
        # 最小块长优先=折叠力度最大;同块长时大 k 优先
        for b in range(min_block, n // 2 + 1):
            for k in range(n // b, 1, -1):
                p = n - k * b
                if p < 0:
                    continue
                if text[p : p + b] * k == text[p:]:
                    return collapse_repeated_text(
                        text[:p] + text[p : p + b], min_sentence_len
                    )
    kept: list[str] = []
    for seg in _SENT_SPLIT_RE.split(text):
        if len(seg) >= min_sentence_len and kept and seg == kept[-1]:
            continue
        kept.append(seg)
    return "".join(kept)


class QuestionRow(TypedDict):
    question: str
    id: int


def dedupe_questions(
    rows: list[QuestionRow],
) -> tuple[list[QuestionRow], dict[int, int]]:
    """按归一化后文本判重:保留首条,返回(唯一行, 首条id→出现次数)。"""
    seen: dict[str, int] = {}
    counts: dict[int, int] = {}
    unique: list[QuestionRow] = []
    for row in rows:
        key = normalize_text(row["question"])
        if key in seen:
            counts[seen[key]] += 1
            continue
        seen[key] = row["id"]
        counts[row["id"]] = 1
        unique.append(row)
    return unique, counts
