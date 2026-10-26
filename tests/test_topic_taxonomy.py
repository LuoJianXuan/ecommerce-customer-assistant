"""ch10 术语表单元测试:17 类权威表、边界说明、标签校验与向量互转。"""

import pytest

from app.topics.taxonomy import (
    N_TOPICS,
    TOPIC_NAMES,
    TOPICS,
    labels_to_vector,
    taxonomy_prompt_block,
    validate_labels,
    vector_to_labels,
)


class TestTaxonomyTable:
    def test_17_topics_unique_ordered(self):
        assert N_TOPICS == 17
        assert len(TOPICS) == len(TOPIC_NAMES) == 17
        assert len(set(TOPIC_NAMES)) == 17

    def test_lead_four_and_other_last(self):
        """四大类领头、其他殿后(用户定死的顺序)。"""
        assert TOPIC_NAMES[:4] == ("退换货", "物流", "尺码", "发票")
        assert TOPIC_NAMES[-1] == "其他"

    def test_every_topic_has_boundary_sentence(self):
        for t in TOPICS:
            assert t.name
            assert t.boundary.strip(), f"{t.name} 缺边界说明"
            assert len(t.boundary) <= 80, f"{t.name} 边界说明应控制在一句话"

    def test_user_neighbor_rules_verbatim(self):
        """用户原话的近邻规则必须落进对应类目的边界说明。"""
        by_name = {t.name: t.boundary for t in TOPICS}
        assert "运费管钱" in by_name["运费"] and "物流" in by_name["运费"]
        assert "价保" in by_name["价保"] and "补差价" in by_name["价保"]
        assert "券" in by_name["优惠活动"] and "价保" in by_name["优惠活动"]


class TestValidateLabels:
    def test_valid_labels_sorted_dedup(self):
        assert validate_labels(["尺码", "退换货", "尺码"]) == ["退换货", "尺码"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_labels([])

    def test_unknown_raises_with_name(self):
        with pytest.raises(ValueError, match="物流时效"):
            validate_labels(["退换货", "物流时效"])


class TestVectorRoundtrip:
    def test_roundtrip(self):
        labels = ["尺码", "退换货"]
        vec = labels_to_vector(labels)
        assert len(vec) == N_TOPICS
        assert sum(vec) == 2
        assert vector_to_labels(vec) == ["退换货", "尺码"]  # 权威表顺序

    def test_vector_positions(self):
        vec = labels_to_vector(["其他"])
        assert vec[TOPIC_NAMES.index("其他")] == 1


class TestPromptBlock:
    def test_prompt_contains_all_names_and_boundaries(self):
        block = taxonomy_prompt_block()
        for t in TOPICS:
            assert t.name in block
            assert t.boundary in block

    def test_prompt_adopts_independent_demand_rule(self):
        """抽审门修订:口径必须是「独立诉求」而非字面关键词。"""
        block = taxonomy_prompt_block()
        assert "独立回答或处理" in block
        assert "背景信息" in block and "不单独打标签" in block
        # 用户裁定的 few-shot 示例必须在
        assert "买大了想退" in block
        assert "玄武星会员的跨星球运费" in block
        assert "积分兑换的礼品" in block
        assert "插座插不进去" in block

    def test_prompt_contains_second_review_rules(self):
        """二次抽审 5 条成批规则必须落在 prompt 里。"""
        block = taxonomy_prompt_block()
        assert "独占兜底" in block and "绝不与可归类标签共存" in block
        assert "假设性后续动作" in block
        assert "这个折扣是只针对会员吗" in block
        assert "红包是优惠不是支付" in block
        assert "外包装破了" in block and "修过一次" in block
