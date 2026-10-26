"""ch07 上下文预算（TDD）：窗口倒推公式、70/30 分层、启动自检。

校准测试钉死演示配置三数（验收 2）：滑窗 5650、层1 3954、层2 1695。
"""

import pytest

from app.context_budget import ContextBudget, budget_from_settings, check_budget, compute_budget

DEMO = dict(
    context_window=18000,
    max_output_tokens=2000,
    max_user_input_tokens=2000,
    max_agent_steps=3,
    tool_result_max_tokens=1200,
    rerank_top_k=5,
    system_prompt_tokens=700,
    want_rounds=10,
    steady_per_round=565,
    safety_ratio=0.10,
    summary_inject_segments=3,
    summary_segment_max_tokens=200,
    evidence_per_chunk_tokens=300,
)


def test_demo_config_calibration_pins_three_numbers():
    """验收钉死：演示配置 → 滑窗 5650、层1 3954、层2 1695。"""
    b = compute_budget(**DEMO)
    assert b.history == 5650
    assert b.layer1 == 3954
    assert b.layer2 == 1695
    # 轮数侧(5650)是 min 的取值方：窗口侧必须更宽
    assert b.window_side == 5800


def test_layer_split_is_70_30_with_one_spare():
    b = compute_budget(**DEMO)
    assert b.layer2 == ceil30(b.history)
    assert b.layer1 == b.history - b.layer2 - 1  # 70% 强制在层1，留 1 token 零头


def ceil30(x: int) -> int:
    import math

    return math.ceil(x * 0.3)


def test_window_side_governs_when_tight():
    """窗口紧时由窗口侧决定（min 的另一支）。"""
    tight = dict(DEMO, context_window=12000)  # 窗口侧 5800-6000=... 重新算
    b = compute_budget(**tight)
    expected_window_side = 12000 - (700 + 5 * 300 + 3 * 200 + 2000 + 2000 + 1200) - 3 * 1200
    assert b.window_side == expected_window_side
    assert b.history == min(10 * 565, expected_window_side)
    assert b.history == expected_window_side  # 该配置下窗口侧更小


def test_alarm_when_cannot_fit_one_round():
    tiny = dict(DEMO, context_window=9000)  # 窗口侧将低于一轮稳态
    b = compute_budget(**tiny)
    alarms = check_budget(b, steady_per_round=565)
    assert any("上下文预算不足" in a for a in alarms)
    ok = compute_budget(**DEMO)
    assert check_budget(ok, steady_per_round=565) == []


def test_budget_from_settings_uses_config_fields():
    from app.config import Settings

    s = Settings(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        agent_max_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        ctx_system_prompt_tokens=700,
        ctx_want_rounds=10,
        ctx_steady_per_round=565,
    )
    b = budget_from_settings(s)
    assert (b.history, b.layer1, b.layer2) == (5650, 3954, 1695)


def test_default_config_budget_holds():
    """默认配置正向：窗口侧远大于轮数侧，历史预算=轮数侧。"""
    from app.config import Settings

    b = budget_from_settings(Settings())
    assert b.history == 5650  # 轮数侧封顶
    assert b.window_side > 5650
