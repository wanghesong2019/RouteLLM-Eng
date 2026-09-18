"""运行时配置热更新的单元测试（TDD RED 阶段）—— ConfigStore 部分。

需求（方案文档 4.8 节 / 问题7）：
    下游强弱模型的 base_url / api_key / 模型名可在**不重启网关**的前提下
    编辑并立即生效。

设计要点：
    1. **不可变配置对象 + 原子替换** —— 请求处理时读到的一定是完整一致
       的某个版本，绝不会读到"半新半旧"的组合
    2. **无锁读** —— 请求路径上 O(1) 读取，不引入竞争
    3. **JSON 持久化** —— 容器重建不丢配置
    4. **密钥掩码** —— GET 接口不回显明文

运行：
    pytest tests/test_config_runtime.py -v
"""

import json
import os
import threading

import pytest


@pytest.fixture
def store(tmp_path):
    from routellm.config_runtime.store import RuntimeConfigStore

    return RuntimeConfigStore(str(tmp_path / "runtime_config.json"))


# ------------------------------------------------------------ 数据模型


def test_runtime_config_is_immutable():
    """配置对象必须不可变（frozen）—— 这是原子替换的基础。"""
    from routellm.config_runtime.store import RuntimeConfig

    cfg = RuntimeConfig(
        strong_model="m1", weak_model="m2",
        api_base="http://x", api_key="sk-1",
    )
    with pytest.raises(Exception):
        cfg.api_key = "sk-2"  # type: ignore[misc]


def test_runtime_config_has_required_fields():
    """须含下游 LLM 的完整配置项。"""
    from routellm.config_runtime.store import RuntimeConfig

    fields = set(RuntimeConfig.__dataclass_fields__)
    for f in ("strong_model", "weak_model", "api_base", "api_key"):
        assert f in fields, f"缺少字段 {f}"


# ------------------------------------------------------------ 基本读写


def test_load_returns_initial(store):
    """初始化后应能读到初始配置。"""
    cfg = store.load()
    assert cfg.strong_model is not None
    assert cfg.api_key is not None


def test_update_takes_effect_immediately(store):
    """update 后立即能读到新值（无缓存延迟）。"""
    before = store.load()
    store.update(strong_model="new-strong-model")
    after = store.load()
    assert after.strong_model == "new-strong-model"
    assert after.strong_model != before.strong_model


def test_update_partial_fields(store):
    """只更新指定字段，其余保持不变。"""
    cfg0 = store.load()
    store.update(api_base="http://changed.example/v1")
    cfg1 = store.load()
    assert cfg1.api_base == "http://changed.example/v1"
    assert cfg1.strong_model == cfg0.strong_model
    assert cfg1.weak_model == cfg0.weak_model
    assert cfg1.api_key == cfg0.api_key


def test_update_returns_new_config(store):
    """update 应返回新配置对象。"""
    new = store.update(weak_model="w-new")
    assert new.weak_model == "w-new"
    assert store.load() is new, "load 应返回同一对象（原子替换）"


# ------------------------------------------------------------ 原子性与并发


def test_update_is_atomic_replacement(store):
    """替换应是整体换对象，而非逐字段修改。

    验证方式：持有旧对象的引用，update 后旧对象内容不变
    —— 说明是"换对象"而非"改对象"。
    """
    old = store.load()
    old_strong = old.strong_model
    store.update(strong_model="brand-new")
    assert old.strong_model == old_strong, "旧对象不应被就地修改"
    assert store.load() is not old, "应换成新对象"


def test_concurrent_read_during_update_never_sees_mixed_state(store):
    """并发读取期间不得读到"半新半旧"的组合。

    写线程反复成对更新 (strong, weak)，读线程校验两字段始终匹配
    —— 若实现是逐字段就地修改，会读到不匹配的组合。
    """
    store.update(strong_model="A0", weak_model="B0")

    stop = threading.Event()
    bad = []

    def writer():
        for i in range(300):
            store.update(strong_model=f"A{i+1}", weak_model=f"B{i+1}")
        stop.set()

    def reader():
        while not stop.is_set():
            cfg = store.load()
            # strong 是 A<n> 时 weak 必须是 B<n>
            if cfg.strong_model.startswith("A") and cfg.weak_model.startswith("B"):
                n1 = cfg.strong_model[1:]
                n2 = cfg.weak_model[1:]
                if n1 != n2:
                    bad.append((cfg.strong_model, cfg.weak_model))

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not bad, f"读到不一致的配置组合: {bad[:5]}"


# ------------------------------------------------------------ 持久化


def test_persists_to_disk(store, tmp_path):
    """更新应落盘。"""
    store.update(strong_model="persisted-model")
    # 新实例读同一文件 → 应看到上次的值
    from routellm.config_runtime.store import RuntimeConfigStore

    store2 = RuntimeConfigStore(str(tmp_path / "runtime_config.json"))
    assert store2.load().strong_model == "persisted-model"


def test_load_from_existing_file(tmp_path):
    """已存在的配置文件应被加载。"""
    p = tmp_path / "runtime_config.json"
    p.write_text(json.dumps({
        "strong_model": "from-file-s",
        "weak_model": "from-file-w",
        "api_base": "http://from-file",
        "api_key": "sk-from-file",
    }), encoding="utf-8")

    from routellm.config_runtime.store import RuntimeConfigStore

    s = RuntimeConfigStore(str(p))
    cfg = s.load()
    assert cfg.strong_model == "from-file-s"
    assert cfg.api_base == "http://from-file"


def test_falls_back_to_env_when_no_file(tmp_path, monkeypatch):
    """配置文件不存在时，应从环境变量初始化（兼容既有部署）。"""
    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "env-strong")
    monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "env-weak")
    monkeypatch.setenv("ROUTELLM_API_BASE", "http://env-base")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-env")

    from routellm.config_runtime.store import RuntimeConfigStore

    s = RuntimeConfigStore(str(tmp_path / "not-exist.json"))
    cfg = s.load()
    assert cfg.strong_model == "env-strong"
    assert cfg.api_key == "sk-env"


def test_file_permissions_are_restricted(store, tmp_path):
    """配置文件含密钥，权限须为 600。"""
    store.update(api_key="sk-secret-value")
    p = tmp_path / "runtime_config.json"
    mode = oct(os.stat(p).st_mode & 0o777)
    assert mode == "0o600", f"权限应为 600，实际 {mode}"


def test_corrupt_file_does_not_crash(tmp_path, monkeypatch):
    """配置文件损坏时不应崩溃 —— 回落到环境变量。"""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "fallback-s")

    from routellm.config_runtime.store import RuntimeConfigStore

    s = RuntimeConfigStore(str(p))
    assert s.load().strong_model == "fallback-s"


# ------------------------------------------------------------ 掩码


def test_masked_hides_api_key(store):
    """掩码视图不得包含明文 api_key。

    使用「FAKE-KEY-FOR-TEST」这类显式假值，避免扫描工具把测试数据
    误判为真实凭据（测试假值不应触发开源卫生告警）。
    """
    fake = "FAKE-KEY-FOR-TEST-0000"
    store.update(api_key=fake)
    m = store.masked()
    assert fake not in json.dumps(m), "掩码后不应出现明文"
    assert "api_key" in m


def test_masked_keeps_other_fields(store):
    """掩码只作用于密钥，其他字段原样返回（供前端回显）。"""
    store.update(strong_model="visible-model", api_base="http://visible")
    m = store.masked()
    assert m["strong_model"] == "visible-model"
    assert m["api_base"] == "http://visible"


def test_mask_helper_format():
    """掩码格式：保留首尾少量字符，中间打码。"""
    from routellm.config_runtime.store import mask_secret

    masked = mask_secret("sk-abcdefghijklmn")
    assert masked.startswith("sk-"), "应保留前缀便于辨识"
    assert "***" in masked
    assert "efghij" not in masked, "中间部分应被遮蔽"

    # 短字符串也要安全处理
    assert mask_secret("abc") == "***"
    assert mask_secret("") == ""
