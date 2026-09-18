"""容错相关配置项测试（方案文档 4.3）。

容错默认**关闭**，因此：
1. 不设环境变量时，所有 resilience_* 取默认值且 enabled=False
2. 显式设置时按环境变量解析（含字符串 → bool/int/float 转换）
3. 非法值在**启用时**才 fail-fast（未启用不应因无关参数报错）
"""

from __future__ import annotations

import pytest

from routellm.config import ConfigError, Settings


def _base_env(**extra):
    e = {
        "ROUTELLM_STRONG_MODEL": "openai/strong",
        "ROUTELLM_WEAK_MODEL": "openai/weak",
        "ROUTELLM_API_BASE": "http://x/v1",
        "ROUTELLM_API_KEY": "k",
    }
    e.update(extra)
    return e


def test_resilience_defaults_off():
    """默认关闭 + 默认参数，保证既有部署行为不变。"""
    s = Settings.from_env(_base_env())
    assert s.resilience_enabled is False
    assert s.resilience_failure_threshold == 5
    assert s.resilience_recovery_timeout == 60.0
    assert s.resilience_max_attempts == 3
    s.validate()  # 不应抛错


def test_resilience_enabled_from_env():
    s = Settings.from_env(_base_env(
        ROUTELLM_RESILIENCE_ENABLED="true",
        ROUTELLM_RESILIENCE_FAILURE_THRESHOLD="3",
        ROUTELLM_RESILIENCE_RECOVERY_TIMEOUT="30",
        ROUTELLM_RESILIENCE_MAX_ATTEMPTS="5",
    ))
    assert s.resilience_enabled is True
    assert s.resilience_failure_threshold == 3
    assert s.resilience_recovery_timeout == 30.0
    assert s.resilience_max_attempts == 5
    s.validate()


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_resilience_enabled_bool_variants(val):
    s = Settings.from_env(_base_env(ROUTELLM_RESILIENCE_ENABLED=val))
    assert s.resilience_enabled is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage"])
def test_resilience_disabled_bool_variants(val):
    s = Settings.from_env(_base_env(ROUTELLM_RESILIENCE_ENABLED=val))
    assert s.resilience_enabled is False


def test_resilience_invalid_int_raises():
    with pytest.raises(ConfigError, match="RESILIENCE_FAILURE_THRESHOLD"):
        Settings.from_env(_base_env(ROUTELLM_RESILIENCE_FAILURE_THRESHOLD="abc"))


def test_resilience_invalid_float_raises():
    with pytest.raises(ConfigError, match="RESILIENCE_RECOVERY_TIMEOUT"):
        Settings.from_env(_base_env(ROUTELLM_RESILIENCE_RECOVERY_TIMEOUT="abc"))


def test_resilience_params_ignored_when_disabled():
    """未启用容错时，无关的非法参数不应阻塞启动（最小惊讶原则）。"""
    s = Settings.from_env(_base_env(ROUTELLM_RESILIENCE_FAILURE_THRESHOLD="0"))
    assert s.resilience_enabled is False
    s.validate()  # 未启用 → 不校验


def test_resilience_params_validated_when_enabled():
    """启用后非法参数必须 fail-fast。"""
    s = Settings.from_env(_base_env(
        ROUTELLM_RESILIENCE_ENABLED="true",
        ROUTELLM_RESILIENCE_FAILURE_THRESHOLD="0",
    ))
    with pytest.raises(ConfigError, match="FAILURE_THRESHOLD"):
        s.validate()


def test_resilience_summary_exposes_flag():
    s = Settings.from_env(_base_env(ROUTELLM_RESILIENCE_ENABLED="true"))
    assert s.summary()["resilience_enabled"] is True
