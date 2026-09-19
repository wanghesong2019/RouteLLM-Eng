"""自适应阈值 + 级联前置过滤的配置接线测试（方案文档 3.5）。

后端需求：Settings 需暴露自适应阈值的开关与参数，供 openai_server 的
lifespan 组装 FastPathRouter / AdaptiveThresholdController。
"""

from routellm.config import Settings


def test_defaults_present():
    """未设环境变量时应有一组安全默认值。"""
    s = Settings.from_env({})
    assert hasattr(s, "fast_path_enabled")
    assert hasattr(s, "adaptive_threshold_enabled")
    assert s.adaptive_tau_base == 0.5
    assert s.adaptive_tau_min == 0.35
    assert s.adaptive_tau_max == 0.75
    assert s.adaptive_k_p == 1.0
    assert s.adaptive_budget_tokens_per_min == 0.0
    assert s.adaptive_latency_sla_ms == 0.0
    assert s.adaptive_sample_interval_sec == 5.0
    assert s.adaptive_window_sec == 60.0


def test_from_env_reads_values():
    """环境变量应覆盖默认值（含前缀 ROUTELLM_）。"""
    s = Settings.from_env(
        {
            "ROUTELLM_FAST_PATH_ENABLED": "true",
            "ROUTELLM_ADAPTIVE_THRESHOLD_ENABLED": "1",
            "ROUTELLM_ADAPTIVE_TAU_BASE": "0.6",
            "ROUTELLM_ADAPTIVE_TAU_MIN": "0.4",
            "ROUTELLM_ADAPTIVE_TAU_MAX": "0.8",
            "ROUTELLM_ADAPTIVE_K_P": "2.0",
            "ROUTELLM_ADAPTIVE_BUDGET_TOKENS_PER_MIN": "50000",
            "ROUTELLM_ADAPTIVE_LATENCY_SLA_MS": "3000",
            "ROUTELLM_ADAPTIVE_SAMPLE_INTERVAL_SEC": "2.5",
            "ROUTELLM_ADAPTIVE_WINDOW_SEC": "120",
        }
    )
    assert s.fast_path_enabled is True
    assert s.adaptive_threshold_enabled is True
    assert s.adaptive_tau_base == 0.6
    assert s.adaptive_tau_min == 0.4
    assert s.adaptive_tau_max == 0.8
    assert s.adaptive_k_p == 2.0
    assert s.adaptive_budget_tokens_per_min == 50000.0
    assert s.adaptive_latency_sla_ms == 3000.0
    assert s.adaptive_sample_interval_sec == 2.5
    assert s.adaptive_window_sec == 120.0


def test_disabled_by_default_but_switchable():
    """级联与自适应阈值默认开启（交付即生效），可用环境变量显式关闭。"""
    s = Settings.from_env({})
    assert s.adaptive_threshold_enabled is True
    assert s.fast_path_enabled is True

    off = Settings.from_env(
        {
            "ROUTELLM_ADAPTIVE_THRESHOLD_ENABLED": "false",
            "ROUTELLM_FAST_PATH_ENABLED": "false",
        }
    )
    assert off.adaptive_threshold_enabled is False
    assert off.fast_path_enabled is False
