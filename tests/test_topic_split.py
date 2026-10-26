"""ch10 T5 单元测试:多标签贪心分层抽样 + 同义词增强(纯函数)。"""

import random

from app.topics.augment import SYNONYM_PAIRS, synonym_augment
import pytest

from app.topics.splitting import (
    deduplicate_rows,
    normalized_overlap,
    normalized_text_key,
    stratified_split,
)


def _rows(n: int = 200, labels_pool=(("退换货",), ("物流",), ("退换货", "尺码"), ("支付",))):
    rng = random.Random(7)
    return [
        {"text": f"样本句子第{i}号", "labels": list(labels_pool[i % len(labels_pool)])}
        for i in range(n)
    ] + [{"text": f"稀有句{i}", "labels": ["评价"]} for i in range(2)]  # 稀有标签只进训练集


class TestStratifiedSplit:
    def test_sizes_approx_80_10_10(self):
        rows = _rows()
        train, dev, test = stratified_split(rows, ratios=(0.8, 0.1, 0.1), seed=42)
        n = len(rows)
        assert abs(len(train) - 0.8 * n) <= 0.02 * n
        assert abs(len(dev) - 0.1 * n) <= 0.02 * n
        assert abs(len(test) - 0.1 * n) <= 0.02 * n

    def test_disjoint_and_complete(self):
        rows = _rows()
        train, dev, test = stratified_split(rows, seed=42)
        texts = [r["text"] for r in train] + [r["text"] for r in dev] + [r["text"] for r in test]
        assert len(texts) == len(rows) == len(set(texts))  # 无泄漏、无丢失

    def test_every_common_label_in_all_sets(self):
        rows = _rows()
        train, dev, test = stratified_split(rows, seed=42)
        for label in ("退换货", "物流", "尺码", "支付"):
            assert any(label in r["labels"] for r in train)
            assert any(label in r["labels"] for r in dev)
            assert any(label in r["labels"] for r in test)

    def test_rare_label_all_in_train(self):
        rows = _rows()
        train, dev, test = stratified_split(rows, seed=42)
        assert sum(1 for r in train if "评价" in r["labels"]) == 2
        assert not any("评价" in r["labels"] for r in dev + test)

    def test_deterministic_with_seed(self):
        rows = _rows()
        a = stratified_split(rows, seed=42)
        b = stratified_split(rows, seed=42)
        assert [r["text"] for r in a[0]] == [r["text"] for r in b[0]]


class TestNormalizedDeduplication:
    def test_key_ignores_width_whitespace_case_and_punctuation(self):
        assert normalized_text_key("ＡＢＣ 客服几点下班？") == normalized_text_key("abc客服几点下班")

    def test_reviewed_duplicate_is_preferred(self):
        rows = [
            {"text": "运费谁出?", "labels": ["运费"], "primary": "运费"},
            {
                "text": "运费谁出",
                "labels": ["运费"],
                "primary": "运费",
                "review_overridden": True,
            },
        ]
        kept, removed = deduplicate_rows(rows)
        assert removed == 1
        assert kept[0]["review_overridden"] is True

    def test_conflicting_duplicate_labels_fail_closed(self):
        rows = [
            {"text": "包裹在哪?", "labels": ["物流"]},
            {"text": "包裹在哪", "labels": ["运费"]},
        ]
        with pytest.raises(ValueError, match="标签冲突"):
            deduplicate_rows(rows)

    def test_normalized_overlap_catches_punctuation_variants(self):
        left = [{"text": "怎么退款?"}]
        right = [{"text": "怎么退款"}]
        assert len(normalized_overlap(left, right)) == 1


class TestSynonymAugment:
    def test_applies_within_class_synonym(self):
        v = synonym_augment("退货运费谁出啊", seed=1)
        assert v is not None and v != "退货运费谁出啊"
        assert "运费" not in v or "邮费" in v or "退了" in v  # 换成了同义表达

    def test_no_match_returns_none(self):
        assert synonym_augment("你好呀", seed=1) is None

    def test_all_pairs_within_class(self):
        """安全护栏:同义对照表每对必须落在同一类目词域内(防止替换改变主题)。"""
        allowed = {
            ("退货", "退了"), ("换货", "调换"), ("退款", "退钱"), ("运费", "邮费"),
            ("快递", "包裹"), ("订单", "单子"), ("发货", "发件"), ("优惠", "折扣"),
            ("保修", "质保"), ("库存", "现货"), ("会员", "VIP"), ("账号", "账户"),
        }
        assert set(SYNONYM_PAIRS) == allowed
