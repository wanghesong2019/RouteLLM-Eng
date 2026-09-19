"""AdaptiveThresholdController 测试（方案文档 3.2）。

方向二：静态阈值 τ → 按实时成本/延迟指标动态调节的 τ(t) ∈ [τ_min, τ_max]，
外加硬质量防线（预算耗尽 + 高难请求 → 背压而非降级）。
"""

from unittest.mock import AsyncMock

import pytest

from routellm.routers.adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdController,
)


class TestAdaptiveThreshold:
    def test_initial_tau_is_base(self):
        """初始阈值应等于 tau_base。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_base=0.5), metrics_store=None
        )
        assert ctrl.get_effective_threshold() == 0.5

    def test_tau_clamped_to_max(self):
        """成本超标 → τ 上调，但不超过 tau_max。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                tau_base=0.5, tau_min=0.35, tau_max=0.75,
                k_p=10.0, budget_tokens_per_min=100,
            ),
            metrics_store=None,
        )
        ctrl._cost_rate = 10000  # 远超预算
        ctrl._update_threshold_sync_for_test()
        assert ctrl.get_effective_threshold() == 0.75

    def test_tau_clamped_to_min(self):
        """成本远低于预算 → τ 下调，但被钳到 tau_min。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                tau_base=0.5, tau_min=0.35, tau_max=0.75,
                k_p=10.0, budget_tokens_per_min=1000,
            ),
            metrics_store=None,
        )
        ctrl._cost_rate = 0
        ctrl._update_threshold_sync_for_test()
        assert ctrl.get_effective_threshold() == 0.35

    def test_hard_guardrail_high_winrate_forces_backpressure(self):
        """s ≥ τ_max 且预算耗尽 → 触发背压（不以次充好）。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                tau_base=0.5, tau_min=0.35, tau_max=0.75,
                budget_tokens_per_min=100,
            ),
            metrics_store=None,
        )
        ctrl._cost_rate = 200  # 超预算
        assert ctrl.should_backpressure(0.8) is True
        assert ctrl.should_backpressure(0.5) is False

    def test_hard_guardrail_forces_strong_within_threshold(self):
        """win_rate ≥ τ_max → 无论动态阈值多低都判强（硬质量防线）。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_base=0.5), metrics_store=None
        )
        ctrl._effective_tau = 0.1  # 阈值被压到很低
        assert ctrl.decide(0.8) == "strong"
        assert ctrl.decide(0.2) == "weak"

    def test_no_budget_no_backpressure(self):
        """未设预算时不应触发背压。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(budget_tokens_per_min=0), metrics_store=None
        )
        assert ctrl.should_backpressure(0.99) is False

    def test_strong_floor_region_below_tau_min(self):
        """s < τ_min → 稳定走弱（安全降本区）。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_min=0.35, tau_max=0.75), metrics_store=None
        )
        assert ctrl.decide(0.2) == "weak"

    @pytest.mark.asyncio
    async def test_update_reads_from_store(self):
        """阈值更新应从 MetricsStore 读取窗口统计。"""
        mock_store = AsyncMock()
        mock_store.window_stats = AsyncMock(
            return_value={
                "tokens_per_min": 200,  # 超预算
                "latency_p90": 500,
                "strong_count": 5,
                "weak_count": 5,
                "total_count": 10,
                "total_cost_usd": 0.5,
            }
        )
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                tau_base=0.5, tau_min=0.35, tau_max=0.75,
                k_p=1.0, budget_tokens_per_min=100,
            ),
            metrics_store=mock_store,
        )
        await ctrl._update_threshold()
        assert ctrl.get_effective_threshold() > 0.5

    @pytest.mark.asyncio
    async def test_sampler_updates_threshold_from_metrics(self):
        """自包含采集器：metrics_provider 的指标应驱动 τ 上调。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                tau_base=0.5, budget_tokens_per_min=100, sample_interval_sec=0.01
            ),
            metrics_provider=lambda since: {
                "tokens_per_min": 1000, "latency_p90": 100,
                "strong_count": 1, "weak_count": 1, "total_count": 2,
                "total_cost_usd": 0.1,
            },
        )
        await ctrl.start()
        import asyncio as _asyncio

        await _asyncio.sleep(0.05)
        await ctrl.stop()
        assert ctrl.get_effective_threshold() > 0.5

    def test_get_status_exposes_tau_and_signals(self):
        """状态视图供 Dashboard / 验收使用。"""
        ctrl = AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_base=0.5, budget_tokens_per_min=100),
            metrics_store=None,
        )
        st = ctrl.get_status()
        assert st["enabled"] is True
        assert st["effective_tau"] == 0.5
        assert st["tau_max"] == 0.75
        assert st["budget_tokens_per_min"] == 100
