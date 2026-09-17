"""Dashboard 配置转发层的测试（TDD RED 阶段）。

方案 B：**配置存网关，Dashboard 通过网关 API 转发**。
    Dashboard 只做 UI + 转发，**不持有配置真相**。

转发链路：
    浏览器 → Dashboard(:8092) /api/config → 网关(:6060) /api/config

为何不在 Dashboard 侧直接改配置：
    - 配置归网关所有（single source of truth）
    - 面板保持"只读挂载 metrics"的干净边界，不持有写入权限
    - 转发层顺带可做审计与错误归一化

配置来源（环境变量）：
    ROUTELLM_GATEWAY_URL  网关地址，默认 http://host.docker.internal:6060
    ROUTELLM_GATEWAY_API_KEY  调用网关的 key（Bearer）

运行：
    pytest tests/test_dashboard_config_proxy.py -v
"""

import asyncio
import json

import pytest


@pytest.fixture
def proxy(monkeypatch):
    import routellm.monitoring.dashboard.config_proxy as m

    monkeypatch.setenv("ROUTELLM_GATEWAY_URL", "http://gw-test:6060")
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-gw-test-key")
    m.reset()
    return m


# ------------------------------------------------------------ 配置读取


def test_gateway_url_from_env(proxy, monkeypatch):
    assert proxy.get_gateway_url() == "http://gw-test:6060"


def test_gateway_url_default(proxy, monkeypatch):
    monkeypatch.delenv("ROUTELLM_GATEWAY_URL", raising=False)
    assert proxy.get_gateway_url() == "http://host.docker.internal:6060"


def test_gateway_url_trailing_slash_stripped(proxy, monkeypatch):
    monkeypatch.setenv("ROUTELLM_GATEWAY_URL", "http://gw:6060/")
    assert proxy.get_gateway_url() == "http://gw:6060"


def test_auth_header_includes_bearer_key(proxy):
    h = proxy.auth_header()
    import os as _os
    assert h["Authorization"] == "Bearer " + _os.environ["ROUTELLM_GATEWAY_API_KEY"]


def test_auth_header_without_key(proxy, monkeypatch):
    """未配 key 时不带 Authorization（网关侧未启用鉴权时可用）。"""
    monkeypatch.delenv("ROUTELLM_GATEWAY_API_KEY", raising=False)
    h = proxy.auth_header()
    assert "Authorization" not in h


# ------------------------------------------------------------ 转发行为


def test_forward_get_config(proxy, monkeypatch):
    """GET 转发应返回网关的掩码配置。"""

    async def fake_request(method, path, body=None, timeout=10.0):
        assert method == "GET"
        assert path == "/api/config"
        return 200, {
            "strong_model": "s", "weak_model": "w",
            "api_base": "http://x", "api_key": "sk-***abcd",
            "editable_fields": ["strong_model"],
        }

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_get_config())
    assert res["strong_model"] == "s"
    assert res.get("proxy_error") is None


def test_forward_put_config(proxy, monkeypatch):
    """PUT 转发应把更新体传给网关。"""
    seen = {}

    async def fake_request(method, path, body=None, timeout=10.0):
        seen["method"] = method
        seen["path"] = path
        seen["body"] = body
        return 200, {"ok": True, "changed": ["api_base"]}

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_put_config({"api_base": "http://new"}))
    assert seen["method"] == "PUT"
    assert seen["body"] == {"api_base": "http://new"}
    assert res["ok"] is True


def test_forward_verify(proxy, monkeypatch):
    """连通性预检转发。"""

    async def fake_request(method, path, body=None, timeout=10.0):
        assert path == "/api/config/verify"
        return 200, {"ok": True, "detail": "200 OK"}

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_verify())
    assert res["ok"] is True


# ------------------------------------------------------------ 错误处理


def test_gateway_unreachable_returns_error_not_raise(proxy, monkeypatch):
    """网关不可达时返回结构化错误（面板不应崩）。"""

    async def fake_request(method, path, body=None, timeout=10.0):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_get_config())
    assert res.get("proxy_error") is not None
    assert "connection refused" in res["proxy_error"] or "ConnectionError" in res["proxy_error"]


def test_gateway_401_propagates(proxy, monkeypatch):
    """网关返回 401 时应明确告知（key 配置错误）。"""

    async def fake_request(method, path, body=None, timeout=10.0):
        return 401, {"error": {"message": "Incorrect API key"}}

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_get_config())
    assert res.get("proxy_error") is not None
    assert "401" in res["proxy_error"]


def test_gateway_5xx_returns_error(proxy, monkeypatch):
    async def fake_request(method, path, body=None, timeout=10.0):
        return 500, {"detail": "boom"}

    monkeypatch.setattr(proxy, "_request", fake_request)
    res = asyncio.run(proxy.forward_get_config())
    assert res.get("proxy_error") is not None


# ------------------------------------------------------------ 路由注册


def test_proxy_routes_registered(proxy):
    paths = {getattr(r, "path", None) for r in proxy.router.routes}
    for p in ("/api/config", "/api/config/verify"):
        assert p in paths, f"缺少路由 {p}"


def test_config_page_route_exists():
    """配置页面路由须存在（前端页签）。"""
    import routellm.monitoring.dashboard.app as m

    paths = {getattr(r, "path", None) for r in m.router.routes}
    assert "/config" in paths, "缺少配置页面路由 /config"
