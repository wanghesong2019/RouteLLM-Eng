"""配置 API 的强弱分侧行为（TDD RED 阶段）。

对应 Dashboard 反馈的三个问题在 **API 层** 的落地：

1. 强弱各自可编辑 base_url / api_key（GET 回显 + PUT 局部更新）
2. 模型名以**原始名**存储/回显（前缀由 Controller 拼）
3. verify 按 tier 分开测试，且响应明示"测的是哪一侧"

运行：
    pytest tests/test_config_per_tier_api.py -v
"""

import asyncio

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    from routellm.config_runtime.store import RuntimeConfigStore

    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "s-init")
    monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "w-init")
    monkeypatch.setenv("ROUTELLM_API_BASE", "https://global.example.com/v1")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-global-key-secret")
    return RuntimeConfigStore(str(tmp_path / "rc.json"))


@pytest.fixture
def api(store):
    import routellm.config_runtime.api as m

    m.set_store(store)
    return m


# ------------------------------------------------- GET：分侧字段回显


def test_editable_fields_include_per_tier(api):
    """可编辑字段须含强弱各自的 base_url / api_key。"""
    res = asyncio.run(api.get_config())
    for f in (
        "strong_model", "weak_model",
        "api_base", "api_key",
        "strong_api_base", "strong_api_key",
        "weak_api_base", "weak_api_key",
    ):
        assert f in res["editable_fields"], f"缺少可编辑字段 {f}"


def test_get_masks_all_keys(api):
    """三处密钥（全局 / 强 / 弱）都不得回显明文。"""
    res = asyncio.run(api.get_config())
    for field in ("api_key", "strong_api_key", "weak_api_key"):
        assert "secret" not in res.get(field, ""), f"{field} 泄漏明文"


# ------------------------------------------------- PUT：分侧局部更新


def test_put_sets_per_tier_credentials(api, store):
    """PUT 可单独设置某一侧的 base_url / key。"""
    from routellm.config_runtime.api import ConfigUpdate

    res = asyncio.run(
        api.put_config(
            ConfigUpdate(
                strong_api_base="https://strong.example.com/v1",
                strong_api_key="sk-strong-key",
            )
        )
    )
    assert res["ok"] is True
    assert store.effective_api_base("strong") == "https://strong.example.com/v1"
    assert store.effective_api_key("strong") == "sk-strong-key"
    # 弱侧未动 → 仍回落全局
    assert store.effective_api_base("weak") == "https://global.example.com/v1"


def test_put_unknown_field_rejected(api):
    """未知字段应被拒绝（防拼写错误静默失效）。"""
    from pydantic import ValidationError

    from routellm.config_runtime.api import ConfigUpdate

    with pytest.raises(ValidationError):
        ConfigUpdate(strong_api_bse="typo")  # 拼错


def test_put_empty_string_clears_per_tier(api, store):
    """传空字符串 = 清空该侧覆盖（回落全局），与 None（不修改）不同。"""
    from routellm.config_runtime.api import ConfigUpdate

    store.update(strong_api_base="https://strong.example.com/v1")
    assert store.effective_api_base("strong") == "https://strong.example.com/v1"

    asyncio.run(api.put_config(ConfigUpdate(strong_api_base="")))
    # 清空后回落全局
    assert store.effective_api_base("strong") == "https://global.example.com/v1"


def test_put_single_model_does_not_error(api, store):
    """只填一个模型名（另一档置空）→ 更新成功，不报错。"""
    from routellm.config_runtime.api import ConfigUpdate

    res = asyncio.run(api.put_config(ConfigUpdate(weak_model="")))
    assert res["ok"] is True
    pair = store.effective_model_pair()
    assert pair.strong == "s-init" and pair.weak == "s-init"


# ------------------------------------------------- verify：分侧测试


def test_verify_accepts_tier(api, monkeypatch):
    """verify 支持 tier 参数，按侧测试。"""
    calls = []

    async def fake_probe(base, key, model, timeout=5.0):
        calls.append({"base": base, "key": key, "model": model})
        return True, "HTTP 200"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)

    res = asyncio.run(api.verify_connectivity(tier="weak"))
    assert res["ok"] is True
    assert res["tier"] == "weak"
    # 测的是弱侧模型
    assert calls[0]["model"] == "w-init"
    assert calls[0]["base"] == "https://global.example.com/v1"


def test_verify_strong_and_weak_use_own_credentials(api, store, monkeypatch):
    """强弱各测各的凭据（各自配置对各自生效）。"""
    calls = []

    async def fake_probe(base, key, model, timeout=5.0):
        calls.append({"base": base, "key": key, "model": model})
        return True, "HTTP 200"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)
    store.update(
        strong_api_base="https://strong.example.com/v1", strong_api_key="sk-s",
        weak_api_base="https://weak.example.com/v1", weak_api_key="sk-w",
    )

    asyncio.run(api.verify_connectivity(tier="strong"))
    asyncio.run(api.verify_connectivity(tier="weak"))

    assert calls[0] == {"base": "https://strong.example.com/v1", "key": "sk-s", "model": "s-init"}
    assert calls[1] == {"base": "https://weak.example.com/v1", "key": "sk-w", "model": "w-init"}


def test_verify_response_names_what_was_probed(api, monkeypatch):
    """响应须明示测的是哪一侧 / 哪个模型 / 哪个地址（消除歧义）。"""
    async def fake_probe(base, key, model, timeout=5.0):
        return True, "HTTP 200"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)

    res = asyncio.run(api.verify_connectivity(tier="strong"))
    assert res["probed"]["tier"] == "strong"
    assert res["probed"]["model"] == "s-init"
    assert res["probed"]["api_base"] == "https://global.example.com/v1"
    assert "***" in res["probed"]["api_key"]
    # 明确能力边界
    assert "模型名" in res["note"]


def test_verify_reports_source_of_credentials(api, store, monkeypatch):
    """响应须说明凭据来源：该侧覆盖 or 顶层兜底。"""
    async def fake_probe(base, key, model, timeout=5.0):
        return True, "HTTP 200"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)

    res = asyncio.run(api.verify_connectivity(tier="weak"))
    assert res["probed"]["source"] == "global", "未配该侧 → 来源应为 global 兜底"

    store.update(weak_api_base="https://weak.example.com/v1")
    res = asyncio.run(api.verify_connectivity(tier="weak"))
    assert res["probed"]["source"] == "weak"


def test_verify_unknown_tier_rejected(api):
    """非法 tier 应返回 422（不静默回落）。"""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        asyncio.run(api.verify_connectivity(tier="medium"))
    assert e.value.status_code == 422


def test_verify_without_tier_still_works(api, monkeypatch):
    """不传 tier 时测全局（向后兼容）。"""
    calls = []

    async def fake_probe(base, key, model, timeout=5.0):
        calls.append({"base": base, "model": model})
        return True, "HTTP 200"

    monkeypatch.setattr(api, "_probe_downstream", fake_probe)

    res = asyncio.run(api.verify_connectivity())
    assert res["tier"] == "global"
    assert calls[0]["base"] == "https://global.example.com/v1"


def test_verify_probe_rejects_missing_key(api, store, monkeypatch):
    """key 为空时预检应失败并给出明确原因（而非发无鉴权请求）。"""
    store.update(api_key="", strong_api_key="", weak_api_key="")
    res = asyncio.run(api.verify_connectivity(tier="strong"))
    assert res["ok"] is False
    assert "api_key" in res["detail"]
