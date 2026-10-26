"""ch10 本地主题推理解码的纯逻辑测试。"""

import json

import pytest

from app.config import Settings
from app.topics.inference import (
    LocalTopicClassifier,
    decode_probabilities,
    validate_model_label_order,
)
from app.topics.taxonomy import TOPIC_NAMES


def test_decode_probabilities_uses_taxonomy_order_and_threshold():
    values = [0.1] * len(TOPIC_NAMES)
    values[1] = 0.41
    values[4] = 0.9

    prediction = decode_probabilities([values], threshold=0.4)[0]

    assert prediction.labels == [TOPIC_NAMES[1], TOPIC_NAMES[4]]
    assert prediction.scores[TOPIC_NAMES[4]] == 0.9


def test_decode_probabilities_allows_empty_labels():
    prediction = decode_probabilities(
        [[0.2] * len(TOPIC_NAMES)], threshold=0.4
    )[0]
    assert prediction.labels == []


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1])
def test_decode_probabilities_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="阈值"):
        decode_probabilities([[0.2] * len(TOPIC_NAMES)], threshold)


def test_decode_probabilities_rejects_wrong_output_dimension():
    with pytest.raises(ValueError, match="输出维度"):
        decode_probabilities([[0.1, 0.2]], threshold=0.4)


def test_classifier_reads_lightweight_metadata_without_loading_weights(tmp_path):
    (tmp_path / "topic_threshold.json").write_text(
        json.dumps({"threshold": 0.4, "model_version": "targeted-v5"}),
        encoding="utf-8",
    )

    classifier = LocalTopicClassifier(Settings(topic_model_path=str(tmp_path)))

    assert classifier.threshold == 0.4
    assert classifier.model_version == "targeted-v5"
    assert classifier._model is None


def test_model_label_order_must_match_taxonomy_exactly():
    validate_model_label_order({i: name for i, name in enumerate(TOPIC_NAMES)})
    wrong = {i: name for i, name in enumerate(TOPIC_NAMES)}
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(RuntimeError, match="id2label"):
        validate_model_label_order(wrong)


def test_corrupt_threshold_metadata_is_reported_as_service_error(tmp_path):
    (tmp_path / "topic_threshold.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="阈值文件"):
        LocalTopicClassifier(Settings(topic_model_path=str(tmp_path)))
