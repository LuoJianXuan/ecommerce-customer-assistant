"""ch10 清洗纯函数单元测试:归一化、脱敏、重复段压缩、按问题去重。"""

from app.topics.cleaning import (
    collapse_repeated_text,
    dedupe_questions,
    desensitize,
    normalize_text,
)


class TestNormalize:
    def test_nfkc_and_fullwidth(self):
        assert normalize_text("你好？这个是全角ＡＢＣ１２３") == "你好?这个是全角ABC123"

    def test_collapse_whitespace(self):
        assert normalize_text("  退货   怎么 \n 办  ") == "退货 怎么 办"


class TestDesensitize:
    def test_phone(self):
        # 人工合成的格式测试夹具，不是用户联系方式；请勿拨打。
        assert desensitize("打我电话13800000000退货") == "打我电话<手机号>退货"

    def test_order_no(self):
        assert desensitize("查下订单SO20261014001") == "查下订单<订单号>"

    def test_amount(self):
        assert desensitize("运费12元还是12.5元") == "运费<金额>还是<金额>"

    def test_normal_text_untouched(self):
        assert desensitize("买大了想退") == "买大了想退"


class TestCollapseRepeated:
    def test_whole_text_repeated_3x(self):
        para = "耳机物流到了没人通知,积分抵扣出错,客服打不通,希望给个说法。"
        assert collapse_repeated_text(para * 3) == para

    def test_repeated_2x(self):
        s = "上次也是拖了好几天才解决,体验非常不好。"
        assert collapse_repeated_text(s * 2) == s

    def test_prefix_plus_block_repeated_3x(self):
        """回归:608 字真实长句=23 字开头+195 字块×3,旧算法漏折(只压到 413)。"""
        head = "再问下你们家无线降噪耳机 E-3 的事,"
        block = "当时下单用的是会员积分抵扣了一部分,物流显示到了之后一直没人通知我去拿,客服电话打不通,希望给个清楚的说法和准确的进度,不要再用模板话术敷衍了事。"
        text = head + block * 3
        assert collapse_repeated_text(text) == head + block

    def test_suffix_period_only(self):
        """无前缀整串自重复走后缀周期分支(块≥16 字),嵌套重复递归折净。"""
        s = "物流卡在转运中心已经整整三天了没有任何更新,"
        assert collapse_repeated_text(s * 4) == s

    def test_block_below_threshold_not_folded(self):
        """块长<16 字的重复视为口语强调,不折叠(护栏边界)。"""
        s = "物流卡住了,"
        assert collapse_repeated_text(s * 3) == s * 3

    def test_normal_text_untouched(self):
        s = "物流太慢了,而且运费还收了12块。"
        assert collapse_repeated_text(s) == s

    def test_no_false_collapse_on_short_repeat(self):
        assert collapse_repeated_text("好，好，谢谢") == "好，好，谢谢"

    def test_doubled_words_not_folded(self):
        """口语叠字(谢谢/呵呵)不是粘贴错误,块长<16 一律不折叠。"""
        assert collapse_repeated_text("谢谢谢谢老板") == "谢谢谢谢老板"


class TestDedupeQuestions:
    def test_dedupe_keeps_first_and_counts(self):
        rows = [
            {"question": "帮我推荐个电饭煲", "id": 11},
            {"question": "帮我推荐个电饭煲 ", "id": 13},  # 空白差异应判重
            {"question": "帮我推荐个电饭煲", "id": 14},
            {"question": "查下会员积分", "id": 22},
        ]
        unique, counts = dedupe_questions(rows)
        assert [r["id"] for r in unique] == [11, 22]
        assert counts == {11: 3, 22: 1}

    def test_fullwidth_judged_same(self):
        rows = [
            {"question": "优惠卷怎么领", "id": 1},
            {"question": "优惠卷怎么领", "id": 2},  # 全角问号 NFKC 后同
        ]
        unique, counts = dedupe_questions(rows)
        assert len(unique) == 1
        assert counts[1] == 2
