"""运行时配置编辑 API 的单元测试（TDD RED 阶段）。

方案 B（用户选定）：**配置存网关，Dashboard 通过网关 API 转发**。
    网关持有配置真相（single source of truth）；
    Dashboard 只是 UI 层，不持有配置。

接口：
    GET  /api/config         掩码视图（不回显密钥明文）
    PUT  /api/config         更新（鉴权 + 可选连通性预检）
    POST /api/config/verify  连通性预检（改 base_url 前试探下游）

安全：
    - 必须走 ApiKeyMiddleware 鉴权（改 key = 改成本中心）
    - GET 不回显明文
    - 变更记审计日志（不记密钥明文）

运行：
    pytest tests/test_config_api.py -v
"""

import asyncio

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    from routellm.config_runtime.store import RuntimeConfigStore

    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "init-strong")
    monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "init-weak")
    monkeypatch.setenv("ROUTELLM_API_BASE", "http://init-base/v1")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-init-key-1234567890")
    return RuntimeConfigStore(str(tmp_path / "rc.json"))


@pytest.fixture
def api(store):
    import routellm.config_runtime.api as m

    m.set_store(store)
    return m


# ------------------------------------------------------------ GET


def test_get_config_returns_masked(api, store):
    """GET 不得回显 api_key 明文。"""
    res = asyncio.run(api.get_config())
    assert res["strong_model"] == "init-strong"
    assert res["api_base"] == "http://init-base/v1"
    # 密钥明文（中间段）不应出现
    assert "key-1234567890" not in res["api_key"], "密钥明文泄漏"
    assert res["api_key"].startswith("sk-")
    assert "***" in res["api_key"], "应为掩码格式"


def test_get_config_includes_editable_fields(api):
    """GET 须列出可编辑字段（供前端渲染表单）。"""
    res = asyncio.run(api.get_config())
    assert "editable_fields" in res
    for f in ("strong_model", "weak_model", "api_base", "api_key"):
        assert f in res["editable_fields"]


# ------------------------------------------------------------ PUT


def test_put_updates_and_takes_effect(api, store):
    """PUT 后配置立即生效（无需重启）。"""
    res = asyncio.run(api.put_config(api.ConfigUpdate(strong_model="new-strong")))
    assert res["ok"] is True
    assert store.load().strong_model == "new-strong"


def test_put_partial_update(api, store):
    """只传部分字段时，其余保持不变。"""
    before = store.load()
    asyncio.run(api.put_config(api.ConfigUpdate(api_base="http://new-base")))
    after = store.load()
    assert after.api_base == "http://new-base"
    assert after.strong_model == before.strong_model
    assert after.api_key == before.api_key


def test_put_empty_payload_is_noop(api, store):
    """空 payload 不应清空配置。"""
    before = store.load()
    res = asyncio.run(api.put_config(api.ConfigUpdate()))
    assert res["ok"] is True
    after = store.load()
    assert after == before


def test_put_response_is_masked(api):
    """PUT 的响应也不应含明文密钥。"""
    res = asyncio.run(api.put_config(api.ConfigUpdate(api_key="sk-new-key-abcdefghij")))
    assert "key-abcdefghij" not in str(res), "响应泄漏密钥"


def test_put_records_audit_without_secret(api, store, caplog):
    """变更应记审计日志，且不包含密钥明文。"""
    import logging

    with caplog.at_level(logging.INFO):
        asyncio.run(api.put_config(api.ConfigUpdate(api_key="sk-new-key-abcdefghij")))

    text = caplog.text
    assert "配置" in text or "config" in text.lower(), "应有变更日志"
    assert "key-abcdefghij" not in text, "日志泄漏密钥明文"


def test_put_rejects_unknown_field(api):
    """未知字段应被拒绝（防止拼写错误静默生效）。"""
    with pytest.raises(Exception):
        api.ConfigUpdate(nonexistent_field="x")


# ------------------------------------------------------------ 连通性预检


def test_verify_endpoint_exists(api):
    """须有连通性预检接口（改 base_url 前试探下游）。"""
    assert hasattr(api, "verify_connectivity")


def test_verify_reports_unreachable(api, monkeypatch):
    """下游不可达时预检应返回 ok=False（而非抛异常）。"""

    async def fake_probe(base, key, model, timeout=5.0):
        return False, "connection refused"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)
    res = asyncio.run(api.verify_connectivity())
    assert res["ok"] is False
    assert "connection refused" in res.get("detail", "")


def test_verify_reports_reachable(api, monkeypatch):
    async def fake_probe(base, key, model, timeout=5.0):
        return True, "200 OK"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)
    res = asyncio.run(api.verify_connectivity())
    assert res["ok"] is True


def test_put_with_verify_blocks_bad_config(api, store, monkeypatch):
    """启用预检时，下游不可达的配置应被拒绝（避免改坏服务）。"""

    async def fake_probe(base, key, model, timeout=5.0):
        return False, "unreachable"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)
    before = store.load()

    res = asyncio.run(
        api.put_config(api.ConfigUpdate(api_base="http://bad-host/v1", verify=True))
    )
    assert res["ok"] is False
    assert store.load() == before, "预检失败时不应写入配置"


def test_put_without_verify_applies_anyway(api, store):
    """未启用预检时（默认），即使下游不可达也写入（保持灵活性）。"""
    res = asyncio.run(api.put_config(api.ConfigUpdate(api_base="http://whatever/v1")))
    assert res["ok"] is True
    assert store.load().api_base == "http://whatever/v1"


# ------------------------------------------------------------ 路由注册


def test_router_registered():
    """API 路由须齐全。"""
    import routellm.config_runtime.api as m

    paths = {getattr(r, "path", None) for r in m.router.routes}
    for p in ("/api/config", "/api/config/verify"):
        assert p in paths, f"缺少路由 {p}"
