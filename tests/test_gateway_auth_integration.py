"""网关鉴权的端到端集成测试（TDD RED→GREEN）。

验证鉴权中间件**真的挂在网关 app 上**并端到端生效 —— 单元测试只验证了
中间件自身逻辑，无法发现"忘记 add_middleware"这类集成遗漏。

运行：
    pytest tests/test_gateway_auth_integration.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """网关 TestClient（关闭监控，避免后台线程/DB 干扰）。"""
    monkeypatch.setenv("ROUTELLM_METRICS_ENABLED", "0")
    monkeypatch.setenv("ROUTELLM_DASHBOARD_ENABLED", "0")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-dummy-downstream")

    import importlib

    import routellm.openai_server as srv

    importlib.reload(srv)
    return TestClient(srv.app, raise_server_exceptions=False)


def test_v1_requires_key_when_enabled(client, monkeypatch):
    """启用鉴权后，/v1/chat/completions 无 key 应 401。"""
    from routellm.monitoring import auth

    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-gw-test")

    r = client.post(
        "/v1/chat/completions",
        json={"model": "router-remote_bert-0.5",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401, f"应 401，实际 {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["error"]["code"] == "invalid_api_key"


def test_v1_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-gw-test")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-wrong"},
        json={"model": "router-remote_bert-0.5",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_v1_accepts_correct_key(client, monkeypatch):
    """正确 key 应通过鉴权（之后可能因下游不可达而 500，但不是 401）。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-gw-test")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-gw-test"},
        json={"model": "router-remote_bert-0.5",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code != 401, "正确 key 不应被鉴权拦下"


def test_health_never_requires_key(client, monkeypatch):
    """/health 免鉴权 —— 容器探针必须能访问。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-gw-test")
    r = client.get("/health")
    assert r.status_code == 200


def test_auth_disabled_when_no_key(client, monkeypatch):
    """未配 key 时业务接口不校验（向后兼容）。"""
    monkeypatch.delenv("ROUTELLM_GATEWAY_API_KEY", raising=False)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "router-remote_bert-0.5",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code != 401, "未启用鉴权时不应返回 401"
