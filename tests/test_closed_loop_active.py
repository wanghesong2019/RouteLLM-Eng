"""闭环"真正跑起来"的验收：默认配置下 τ 会随真实流量变化。

前提（用户要求，2026-09-19）：budget 默认 1400 tok/min（单人交互对话口径），
目标是交付即生效 —— 不是"装好了但没发动"。本测试守住这条。
"""

import asyncio
import time

import pytest

from routellm.config import Settings
from routellm.routers.adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdController,
    WindowMetricsCounter,
)


def test_shipped_default_budget_is_nonzero():
    """出厂默认预算必须非 0，否则闭环永不介入（等于没上线）。"""
    s = Settings.from_env({})
    assert s.adaptive_threshold_enabled is True
    assert s.adaptive_budget_tokens_per_min > 0, (
        "默认预算为 0 —— τ 将恒为 τ_base，自适应闭环不会实际发生作用"
    )


@pytest.mark.asyncio
async def test_closed_loop_engages_above_default_budget():
    """超过默认预算的流量应把 τ 顶到上限，判决随之转向弱模型。"""
    s = Settings.from_env({})
    budget = s.adaptive_budget_tokens_per_min

    counter = WindowMetricsCounter(window_sec=s.adaptive_window_sec)
    # 灌入 3× 预算的窗口消耗
    for _ in range(30):
        counter.record(int(budget * 3 / 30))

    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(
            tau_base=s.adaptive_tau_base, tau_min=s.adaptive_tau_min,
            tau_max=s.adaptive_tau_max, k_p=s.adaptive_k_p,
            budget_tokens_per_min=budget,
            window_sec=s.adaptive_window_sec,
        ),
        metrics_provider=counter.snapshot,
    )
    before = ctrl.get_effective_threshold()
    await ctrl._update_threshold()
    after = ctrl.get_effective_threshold()

    assert after > before, f"超预算未上调 τ（{before} -> {after}）"
    assert after == pytest.approx(s.adaptive_tau_max), "τ 应被钳到上限"


@pytest.mark.asyncio
async def test_closed_loop_relaxes_below_default_budget():
    """低于预算的流量应把 τ 压到下限，判决随之转向强模型（保质量）。"""
    s = Settings.from_env({})

    counter = WindowMetricsCounter(window_sec=s.adaptive_window_sec)
    for _ in range(5):
        counter.record(10)  # 远低于预算

    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(
            tau_base=s.adaptive_tau_base, tau_min=s.adaptive_tau_min,
            tau_max=s.adaptive_tau_max, k_p=s.adaptive_k_p,
            budget_tokens_per_min=s.adaptive_budget_tokens_per_min,
            window_sec=s.adaptive_window_sec,
        ),
        metrics_provider=counter.snapshot,
    )
    await ctrl._update_threshold()
    assert ctrl.get_effective_threshold() == pytest.approx(s.adaptive_tau_min)


@pytest.mark.asyncio
async def test_hard_guardrail_holds_at_default_config():
    """无论 τ 被压到多低，s ≥ τ_max 都必须走强（硬质防线不可被闭环绕过）。"""
    s = Settings.from_env({})
    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(
            tau_base=s.adaptive_tau_base, tau_min=s.adaptive_tau_min,
            tau_max=s.adaptive_tau_max,
            budget_tokens_per_min=s.adaptive_budget_tokens_per_min,
        ),
        metrics_provider=None,
    )
    ctrl._effective_tau = s.adaptive_tau_min  # 故意压到最低
    assert ctrl.decide(s.adaptive_tau_max) == "strong"
    assert ctrl.decide(0.0) == "weak"


@pytest.mark.asyncio
async def test_backpressure_possible_at_default_budget():
    """默认预算下，超 1.5 倍 + 高难请求应能触发背压（防线可达）。"""
    s = Settings.from_env({})
    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(
            tau_max=s.adaptive_tau_max,
            budget_tokens_per_min=s.adaptive_budget_tokens_per_min,
        ),
        metrics_provider=None,
    )
    ctrl._cost_rate = s.adaptive_budget_tokens_per_min * 2  # 超 1.5 倍阈值
    assert ctrl.should_backpressure(s.adaptive_tau_max) is True
