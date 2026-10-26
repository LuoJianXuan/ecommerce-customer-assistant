"""ch07 上下文预算：从模型窗口倒推，不写死常量（教程文案口径）。

公式：
    固定开销 = 系统提示 + 检索证据(top_k×每条) + 梗概预留(注入段数×段上限)
             + 输出预留 + 输入预留 + 安全余量(比例×窗口)
    瞬时峰值 = 步数 × 工具结果上限      （本轮 ReAct 跑起来的最大瞬时占用）
    窗口侧   = 窗口 − 固定开销 − 峰值
    历史预算 = min(想留住轮数 × 每轮稳态, 窗口侧)
    层2 = ceil(30% × 历史预算)；层1 = 历史预算 − 层2 − 1（七成强制给层1，留 1 token 零头）

口径约束：token 计数全程用同一计数器（app/tokens.py），预算与裁剪只改一个则净效果相反。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    history: int  # 滑窗总预算（层1+层2）
    layer1: int
    layer2: int
    window_side: int  # 窗口侧余额（min 的另一支）
    fixed_overhead: int
    peak: int


def compute_budget(
    *,
    context_window: int,
    max_output_tokens: int,
    max_user_input_tokens: int,
    max_agent_steps: int,
    tool_result_max_tokens: int,
    rerank_top_k: int,
    system_prompt_tokens: int,
    want_rounds: int,
    steady_per_round: int,
    safety_ratio: float,
    summary_inject_segments: int,
    summary_segment_max_tokens: int,
    evidence_per_chunk_tokens: int,
) -> ContextBudget:
    fixed = (
        system_prompt_tokens
        + rerank_top_k * evidence_per_chunk_tokens
        + summary_inject_segments * summary_segment_max_tokens
        + max_output_tokens
        + max_user_input_tokens
        + int(safety_ratio * context_window)
    )
    peak = max_agent_steps * tool_result_max_tokens
    window_side = context_window - fixed - peak
    history = max(0, min(want_rounds * steady_per_round, window_side))
    layer2 = math.ceil(history * 0.3)
    layer1 = max(0, history - layer2 - 1)
    return ContextBudget(
        history=history, layer1=layer1, layer2=layer2,
        window_side=window_side, fixed_overhead=fixed, peak=peak,
    )


def budget_from_settings(settings) -> ContextBudget:
    """从 Settings 取数（生产/启动自检入口）。"""
    return compute_budget(
        context_window=settings.model_context_window,
        max_output_tokens=settings.max_output_tokens,
        max_user_input_tokens=settings.max_user_input_tokens,
        max_agent_steps=settings.agent_max_steps,
        tool_result_max_tokens=settings.tool_result_max_tokens,
        rerank_top_k=settings.rerank_top_k,
        system_prompt_tokens=settings.ctx_system_prompt_tokens,
        want_rounds=settings.ctx_want_rounds,
        steady_per_round=settings.ctx_steady_per_round,
        safety_ratio=settings.ctx_safety_ratio,
        summary_inject_segments=settings.ctx_summary_inject_segments,
        summary_segment_max_tokens=settings.ctx_summary_segment_max_tokens,
        evidence_per_chunk_tokens=settings.ctx_evidence_per_chunk_tokens,
    )


def check_budget(budget: ContextBudget, steady_per_round: int) -> list[str]:
    """启动自检：连一轮都装不下就报警（验收 2 的「上下文预算不足」）。"""
    alarms: list[str] = []
    if budget.window_side < steady_per_round or budget.history <= 0:
        alarms.append(
            f"上下文预算不足：窗口侧 {budget.window_side} token < 一轮稳态 {steady_per_round} token"
            f"（固定开销 {budget.fixed_overhead} + 峰值 {budget.peak}），"
            f"请调大 MODEL_CONTEXT_WINDOW 或调小固定开销配置"
        )
    return alarms
