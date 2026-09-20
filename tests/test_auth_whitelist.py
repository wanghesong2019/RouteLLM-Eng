"""鉴权白名单测试（TDD）。

背景：dashboard 经反向代理暴露在公网后，
点击页面导航"运行时配置"（<a href="/config">）时报 401 invalid_api_key。

根因：网关 auth.py 的 WHITELIST_EXACT 里有 "/dashboard"，但没有 "/config"，
而 dashboard 页面的导航链接指向的是 "/config" —— 名称对不上，导致整页
跳转（无法携带 Authorization header）被鉴权中间件拦下。

本测试锁定契约：面板自身的页面路由（/dashboard 与 /config）应等价处理，
都不需要业务 key，否则面板导航不可用。
"""

import pytest


@pytest.fixture(autouse=True)
def _enable_auth(monkeypatch):
    """启用鉴权，否则 requires_auth 一律放行，测试无意义。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "test-key-abc123")


def test_dashboard_page_needs_no_auth():
    """既有契约：/dashboard 免鉴权（回归保护）。"""
    from routellm.monitoring.auth import requires_auth

    assert requires_auth("/dashboard") is False


def test_config_page_needs_no_auth():
    """RED：面板的"运行时配置"页面 /config 也应免鉴权。

    它是 dashboard 自己的页面路由（<a href="/config">），与 /dashboard 同类。
    若返回 True，页面整页跳转必然 401（浏览器导航带不了 Authorization）。
    """
    from routellm.monitoring.auth import requires_auth

    assert requires_auth("/config") is False


def test_business_endpoints_still_require_auth():
    """回归保护：业务接口必须继续要求鉴权，不能因为放宽白名单而失守。"""
    from routellm.monitoring.auth import requires_auth

    for path in ("/v1/chat/completions", "/v1/models"):
        assert requires_auth(path) is True, f"{path} 必须仍然需要鉴权"


def test_unknown_path_still_requires_auth():
    """回归保护：未知路径默认仍需鉴权（白名单是显式枚举，不是通配）。"""
    from routellm.monitoring.auth import requires_auth

    assert requires_auth("/config/secret-admin") is True
