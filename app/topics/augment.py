"""ch10 训练集同义词增强(纯函数,T5)。

安全原则:同义对照表的每一对必须落在同一类目词域内——替换不得改变句子的主题
标签(跨类目近义词如 快递↔物流 属于类目词本身,严禁入表)。只替换首个命中,
保守增强;无适用对返回 None。
"""

import random

SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("退货", "退了"),
    ("换货", "调换"),
    ("退款", "退钱"),
    ("运费", "邮费"),
    ("快递", "包裹"),
    ("订单", "单子"),
    ("发货", "发件"),
    ("优惠", "折扣"),
    ("保修", "质保"),
    ("库存", "现货"),
    ("会员", "VIP"),
    ("账号", "账户"),
)


def synonym_augment(text: str, seed: int) -> str | None:
    """命中同义对则替换首个出现生成变体;无适用对返回 None。"""
    rng = random.Random(seed)
    pairs = [p for p in SYNONYM_PAIRS if p[0] in text and p[1] not in text]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    return text.replace(a, b, 1)
