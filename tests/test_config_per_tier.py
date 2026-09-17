"""强弱模型各自的配置 + 模型名原始形式（TDD RED 阶段）。

背景（Dashboard 反馈的三个问题）
--------------------------------
1. **强弱配置不对等**：弱模型只有模型名一个可编辑项，base_url / api_key
   是网关级全局项，强弱共用。一个网关接两个不同 provider 的拓扑表达不了。
   → 需求：强弱**各自**有 base_url / api_key。
2. **模型名必须带 openai/ 前缀**：用户希望配置**原始模型名**
   （如 `deepseek-ai/DeepSeek-V4-Pro`），前缀由代码在调用层补。
3. **只填一个模型名时应可用**：没填的那一档不能报错，所有请求都走已填的那档。
   （用户原话：\"如果用户只填一个模型名，那后续所有的请求都走这个模型，
     不能路由到没填的那个报错\"）

设计（用户确认的方案）
----------------------
- **模型层兜底**：某一档模型名为空 → 并入已配的那档（谁配用谁）
- **地址层回落**：某侧 api_base / api_key 为空 → 回落**顶层** api_base / api_key
  （顶层保留为全局兜底，向后兼容零破坏）
- **两侧都空** → 校验期 fail-fast，不留到请求期报 400
- **前缀**：存储层存原始名，`downstream_kwargs` 拼 `openai/` 前缀（方案 a）

运行：
    pytest tests/test_config_per_tier.py -v
"""

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    from routellm.config_runtime.store import RuntimeConfigStore

    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "strong-init")
    monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "weak-init")
    monkeypatch.setenv("ROUTELLM_API_BASE", "https://global.example.com/v1")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-global")
    return RuntimeConfigStore(str(tmp_path / "rc.json"))


# ======================================================= 1. 数据模型扩展


def test_runtime_config_has_per_tier_fields():
    """RuntimeConfig 须含强弱各自的 base_url / api_key 字段。"""
    from routellm.config_runtime.store import RuntimeConfig

    fields = set(RuntimeConfig.__dataclass_fields__)
    for f in (
        "strong_model", "weak_model",
        "api_base", "api_key",  # 顶层 = 全局兜底
        "strong_api_base", "strong_api_key",
        "weak_api_base", "weak_api_key",
    ):
        assert f in fields, f"缺少字段 {f}"


def test_per_tier_fields_default_empty():
    """新增的每侧字段默认空（旧调用方不必传）。"""
    from routellm.config_runtime.store import RuntimeConfig

    cfg = RuntimeConfig(
        strong_model="s", weak_model="w",
        api_base="http://g", api_key="sk-g",
    )
    assert cfg.strong_api_base == ""
    assert cfg.weak_api_key == ""


# ======================================================= 2. 地址层回落


def test_effective_base_falls_back_to_global(store):
    """某侧未配 base_url → 用顶层全局值。"""
    cfg = store.load()
    assert store.effective_api_base("strong") == "https://global.example.com/v1"
    assert store.effective_api_base("weak") == "https://global.example.com/v1"
    assert store.effective_api_key("strong") == "sk-global"


def test_effective_base_prefers_per_tier(store):
    """某侧配了 → 用该侧的值，不回落。"""
    store.update(
        strong_api_base="https://strong.example.com/v1",
        strong_api_key="sk-strong",
    )
    assert store.effective_api_base("strong") == "https://strong.example.com/v1"
    assert store.effective_api_key("strong") == "sk-strong"
    # 弱侧未配 → 仍回落全局
    assert store.effective_api_base("weak") == "https://global.example.com/v1"


def test_per_tier_isolation(store):
    """强弱各自配置互不影响（用户核心诉求：各自配置对各自生效）。"""
    store.update(
        strong_api_base="https://a.example.com/v1", strong_api_key="sk-a",
        weak_api_base="https://b.example.com/v1", weak_api_key="sk-b",
    )
    assert store.effective_api_base("strong") == "https://a.example.com/v1"
    assert store.effective_api_base("weak") == "https://b.example.com/v1"
    assert store.effective_api_key("strong") == "sk-a"
    assert store.effective_api_key("weak") == "sk-b"


def test_effective_unknown_tier_raises(store):
    """非法 tier 名应报错（防止拼错静默回落）。"""
    with pytest.raises(ValueError):
        store.effective_api_base("medium")


# ======================================================= 3. 模型层兜底


def test_single_model_strong_only(store):
    """只配强模型 → 弱档并入强档（谁配用谁）。"""
    store.update(weak_model="")
    pair = store.effective_model_pair()
    assert pair.strong == "strong-init"
    assert pair.weak == "strong-init", "只配一个模型时，弱档应并入已配档"


def test_single_model_weak_only(store):
    """只配弱模型 → 强档并入弱档。"""
    store.update(strong_model="")
    pair = store.effective_model_pair()
    assert pair.weak == "weak-init"
    assert pair.strong == "weak-init"


def test_both_models_kept_when_present(store):
    """两侧都有 → 保持原样，不做任何合并。"""
    pair = store.effective_model_pair()
    assert pair.strong == "strong-init"
    assert pair.weak == "weak-init"


def test_both_models_empty_raises(store):
    """两侧都空 → 必须报错（真错误，不可静默）。"""
    store.update(strong_model="", weak_model="")
    with pytest.raises(ValueError):
        store.effective_model_pair()


# ======================================================= 4. 模型名前缀


def test_downstream_kwargs_adds_openai_prefix(store):
    """存原始名 → 调用下游时拼 openai/ 前缀（方案 a）。"""
    from routellm.controller import Controller

    store.update(strong_model="deepseek-ai/DeepSeek-V4-Pro", weak_model="deepseek-ai/DeepSeek-V4-Flash")
    c = Controller(
        routers=[], strong_model="x", weak_model="y",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    kw = c.downstream_kwargs("deepseek-ai/DeepSeek-V4-Pro", tier="strong")
    assert kw["model"] == "openai/deepseek-ai/DeepSeek-V4-Pro"


def test_downstream_kwargs_does_not_double_prefix(store):
    """已带前缀的名字不再重复添加。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="x", weak_model="y",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    kw = c.downstream_kwargs("openai/deepseek-ai/DeepSeek-V4-Pro", tier="strong")
    assert kw["model"] == "openai/deepseek-ai/DeepSeek-V4-Pro"
    assert not kw["model"].startswith("openai/openai/")


def test_downstream_kwargs_uses_per_tier_credentials(store):
    """按 tier 取对应侧的 base_url / api_key（各自配置对各自生效）。"""
    from routellm.controller import Controller

    store.update(
        strong_api_base="https://strong.example.com/v1", strong_api_key="sk-strong",
        weak_api_base="https://weak.example.com/v1", weak_api_key="sk-weak",
    )
    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    kw_s = c.downstream_kwargs("m-strong", tier="strong")
    kw_w = c.downstream_kwargs("m-weak", tier="weak")
    assert kw_s["api_base"] == "https://strong.example.com/v1"
    assert kw_s["api_key"] == "sk-strong"
    assert kw_w["api_base"] == "https://weak.example.com/v1"
    assert kw_w["api_key"] == "sk-weak"


def test_downstream_kwargs_without_tier_falls_back_to_global(store):
    """不传 tier（向后兼容）→ 用顶层配置。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    kw = c.downstream_kwargs("m")
    assert kw["api_base"] == "https://global.example.com/v1"
    assert kw["api_key"] == "sk-global"


# ======================================================= 5. 路由决策兜底


def test_routed_model_never_empty_when_single_model(store):
    """只配一个模型时，路由决策绝不能返回空模型名。"""
    from routellm.controller import Controller

    store.update(strong_model="only-model", weak_model="")
    c = Controller(
        routers=[], strong_model="x", weak_model="y",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    pair = c.live_model_pair()
    assert pair.strong == "only-model"
    assert pair.weak == "only-model", "空档须并入已配档，不能留空导致下游 400"


# ======================================================= 6. 掩码视图


def test_masked_hides_both_keys(store):
    """掩码视图须同时遮蔽顶层与每侧的 key。"""
    store.update(
        strong_api_key="sk-strong-secret-1234",
        weak_api_key="sk-weak-secret-5678",
    )
    m = store.masked()
    for field in ("api_key", "strong_api_key", "weak_api_key"):
        assert "secret" not in m[field], f"{field} 泄漏了明文"
    assert m["api_key_masked"] is True
    # 地址不是密钥，应原样回显
    assert m["api_base"] == "https://global.example.com/v1"


def test_masked_view_reports_single_model_mode(store):
    """掩码视图应能反映\"单模型模式\"（前端据此提示用户）。"""
    store.update(weak_model="")
    m = store.masked()
    assert m["single_model_mode"] is True
    assert m["effective_weak_model"] == "strong-init"
