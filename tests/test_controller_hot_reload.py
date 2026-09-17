"""配置热更新接入 Controller 的测试（TDD RED 阶段）。

核心需求：**改配置后不重启，下一个请求就用新配置。**

改造点（`controller.py`）：
    现状：`self.model_pair` / `self.api_base` / `self.api_key` 在
    `__init__` 时从 Settings 取值后固化，之后不再更新。

    改造：改为从 `RuntimeConfigStore` **按请求现取**，从而支持热更新。

验证方式（关键）：
    1. 改配置 → 同一次进程内下一个请求即用新值（无需重启）
    2. 未配置 store 时行为不变（向后兼容）
    3. 配置对象原子性：请求处理期间读到的一定是完整版本

运行：
    pytest tests/test_controller_hot_reload.py -v
"""

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    from routellm.config_runtime.store import RuntimeConfigStore

    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "s-init")
    monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "w-init")
    monkeypatch.setenv("ROUTELLM_API_BASE", "http://base-init/v1")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-init")
    return RuntimeConfigStore(str(tmp_path / "rc.json"))


def _make_controller():
    """构造最小 Controller（不加载真实路由器）。"""
    from routellm.controller import Controller

    return Controller(
        routers=[],
        strong_model="s-from-args",
        weak_model="w-from-args",
        api_base="http://from-args",
        api_key="sk-args",
    )


# ------------------------------------------------------- 配置来源


def test_controller_accepts_config_store(store):
    """Controller 应可注入 RuntimeConfigStore。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x",
        config_store=store,
    )
    assert c.config_store is store


def test_live_config_reads_from_store(store):
    """注入 store 后，live_config() 应返回 store 里的值。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="ignored", weak_model="ignored",
        api_base="ignored", api_key="ignored", config_store=store,
    )
    cfg = c.live_config()
    assert cfg.strong_model == "s-init"
    assert cfg.api_base == "http://base-init/v1"


def test_live_config_falls_back_to_args_without_store():
    """未注入 store 时，live_config() 回落到构造参数（向后兼容）。"""
    c = _make_controller()
    cfg = c.live_config()
    assert cfg.strong_model == "s-from-args"
    assert cfg.api_base == "http://from-args"


# ------------------------------------------------------- 热更新生效


def test_config_change_takes_effect_without_restart(store):
    """改配置后，下一次读取即为新值（无需重启）。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://old", api_key="sk-old", config_store=store,
    )
    assert c.live_config().api_base == "http://base-init/v1"

    store.update(api_base="http://new-base/v1")

    # 同一个 Controller 实例，未重启
    assert c.live_config().api_base == "http://new-base/v1"


def test_model_pair_reflects_live_config(store):
    """ModelPair 须按请求现取（强弱模型名可热更）。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    assert c.live_model_pair().strong == "s-init"

    store.update(strong_model="s-hot")
    assert c.live_model_pair().strong == "s-hot"


def test_live_config_is_consistent_snapshot(store):
    """一次 live_config() 调用返回的应是完整一致的配置快照。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    store.update(strong_model="A", weak_model="B")
    snap = c.live_config()
    # 同一对象内字段应自洽（原子替换保证）
    assert snap.strong_model == "A" and snap.weak_model == "B"
    # 对象不可变
    with pytest.raises(Exception):
        snap.strong_model = "X"  # type: ignore[misc]


def test_build_request_kwargs_uses_live_config(store):
    """构造下游请求参数时应使用 live 配置（base/key/model）。"""
    from routellm.controller import Controller

    c = Controller(
        routers=[], strong_model="s", weak_model="w",
        api_base="http://x", api_key="sk-x", config_store=store,
    )
    store.update(api_base="http://hot-base/v1", api_key="sk-hot")

    kw = c.downstream_kwargs("s-init")
    assert kw["api_base"] == "http://hot-base/v1"
    assert kw["api_key"] == "sk-hot"
