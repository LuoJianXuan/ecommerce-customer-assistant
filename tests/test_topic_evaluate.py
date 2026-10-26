"""ch10 T7 指标与阈值选择的纯逻辑测试。"""

import numpy as np

from training.evaluate import metric_summary, select_threshold


def test_metric_summary_distinguishes_all_and_supported_label_macro():
    y_true = np.array([[1, 0, 0], [1, 0, 0]])
    y_pred = np.array([[1, 0, 0], [1, 0, 0]])
    metrics = metric_summary(y_true, y_pred)
    assert metrics["macro_f1"] == 0.3333
    assert metrics["supported_macro_f1"] == 1.0


def test_threshold_is_selected_only_from_supplied_dev_predictions():
    y_true = np.array([[1, 0], [0, 1]])
    probs = np.array([[0.42, 0.10], [0.10, 0.42]])
    threshold, sweep = select_threshold(y_true, probs, candidates=[0.4, 0.5])
    assert threshold == 0.4
    assert len(sweep) == 2


def test_threshold_tie_prefers_value_closest_to_half():
    y_true = np.array([[1, 0]])
    probs = np.array([[0.9, 0.1]])
    threshold, _ = select_threshold(y_true, probs, candidates=[0.3, 0.5])
    assert threshold == 0.5
