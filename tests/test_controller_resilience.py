"""Controller 容错接入测试（方案文档 4.3 步骤 5）。

改造前的缺陷：`Controller.acompletion()` 裸调 `litellm.acompletion()`，
下游任何一次抖动都会直接把异常抛给客户端。

改造后应满足：
1. 强模型调用失败 → 自动降级到弱模型，并标记响应为降级
2. 强弱都失败 → 查缓存兜底
3. 全部耗尽 → 抛 FallbackExhaustedError（server 层转 503）
4. 正常路径行为不变（不降级、不改变返回值）
5. 熔断器按路由器名和模型名分别实例化（互不干扰）

这里用 monkeypatch 替换 litellm 调用，避免真实网络。
"""

from __future__ import annotations

import asyncio

import pytest

from routellm.controller import Controller
from routellm.resilience import CircuitState, FallbackExhaustedError


class _FakeResp:
    """最小可用的 litellm 响应替身。"""

    def __init__(self, text: str = "ok", model: str = "m") -> None:
        self.text = text
        self.model = model

    def model_dump(self):
        return {"choices": [{"message": {"content": self.text}}], "model": self.model}


def _make_controller(**kw):
    """构造 Controller，并注册一个假路由器让阈值校验通过。

    注意：`acompletion` 会先调 `_validate_router_threshold`，路由器必须在
    `self.routers` 里，否则直接抛 RoutingError、根本走不到下游调用 ——
    这正是本测试文件早期版本全部失败的原因。
    """
    c = Controller(
        routers=["random"],
        strong_model="strong-model",
        weak_model="weak-model",
        api_base="http://x/v1",
        api_key="k",
        **kw,
    )
    return c


@pytest.mark.asyncio
async def test_acompletion_falls_back_to_weak_on_strong_failure(monkeypatch):
    """强模型失败 → 降级到弱模型（降级链②）。"""
    c = _make_controller()
    calls: list[str] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        # litellm 收到的 model 已带 provider 前缀
        if "strong-model" in kwargs["model"]:
            raise TimeoutError("strong down")
        return _FakeResp("weak-answer")

    monkeypatch.setattr("routellm.controller.acompletion", fake_acompletion)
    # 注意：acompletion 走**异步**路由路径（方案文档 4.4），须 patch 异步版
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)
    c.resilience_enabled = True
    c.resilience_max_attempts = 1

    res = await c.acompletion(model="router-random-0.5",
                              messages=[{"role": "user", "content": "hi"}])
    assert isinstance(res, _FakeResp)
    assert res.text == "weak-answer"
    assert c.last_downgraded is True, "降级必须可观测"


@pytest.mark.asyncio
async def test_acompletion_normal_path_not_downgraded(monkeypatch):
    """正常路径：强模型成功 → 不标记降级。"""
    c = _make_controller()

    async def fake_acompletion(**kwargs):
        return _FakeResp("strong-answer")

    monkeypatch.setattr("routellm.controller.acompletion", fake_acompletion)
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)
    c.resilience_enabled = True

    res = await c.acompletion(model="router-random-0.5",
                              messages=[{"role": "user", "content": "hi"}])
    assert res.text == "strong-answer"
    assert c.last_downgraded is False


@pytest.mark.asyncio
async def test_acompletion_exhausted_raises_fallback_error(monkeypatch):
    """强弱都失败且无缓存 → FallbackExhaustedError（降级链④）。"""
    c = _make_controller()

    async def always_timeout(**kwargs):
        raise TimeoutError("down")

    monkeypatch.setattr("routellm.controller.acompletion", always_timeout)
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)
    c.resilience_enabled = True
    c.resilience_max_attempts = 1  # 不重试，单次失败即降级

    with pytest.raises(FallbackExhaustedError) as e:
        await c.acompletion(model="router-random-0.5",
                            messages=[{"role": "user", "content": "hi"}])
    assert e.value.retry_after == 30


@pytest.mark.asyncio
async def test_acompletion_uses_cache_when_both_fail(monkeypatch):
    """强弱都失败 → 缓存兜底（降级链③）。"""
    c = _make_controller()
    cache_store: dict = {}

    class _Cache:
        def get(self, k):
            return cache_store.get(k)

        def set(self, k, v, ttl=None):
            cache_store[k] = v

    c.resilience_cache = _Cache()

    async def always_timeout(**kwargs):
        raise TimeoutError("down")

    monkeypatch.setattr("routellm.controller.acompletion", always_timeout)
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)
    c.resilience_enabled = True
    c.resilience_max_attempts = 1

    # 先塞一条历史响应（模拟之前成功过）
    from routellm.cache.keys import result_key
    msgs = [{"role": "user", "content": "hi"}]
    import json
    key = result_key(json.dumps(msgs, sort_keys=True, ensure_ascii=False))
    cache_store[key] = _FakeResp("cached-answer")

    res = await c.acompletion(model="router-random-0.5", messages=msgs)
    assert res.text == "cached-answer"
    assert c.last_downgraded is True


@pytest.mark.asyncio
async def test_breakers_are_per_router_and_per_model(monkeypatch):
    """熔断器应按路由器名与模型名分别实例化，互不干扰。"""
    c = _make_controller()
    c.resilience_enabled = True

    b1 = c._get_breaker("router_a", "strong")
    b2 = c._get_breaker("router_b", "strong")
    b3 = c._get_breaker("router_a", "weak")

    assert b1 is not b2, "不同路由器不应共用熔断器"
    assert b1 is not b3, "同一路由器的强弱侧不应共用熔断器"
    assert c._get_breaker("router_a", "strong") is b1, "同一 key 应复用实例"


@pytest.mark.asyncio
async def test_resilience_disabled_keeps_legacy_behavior(monkeypatch):
    """未启用容错时保持原行为（向后兼容）：异常直接抛出。"""
    c = _make_controller()
    # 默认不应启用（避免改变既有部署行为）
    assert getattr(c, "resilience_enabled", False) is False

    async def always_timeout(**kwargs):
        raise TimeoutError("down")

    monkeypatch.setattr("routellm.controller.acompletion", always_timeout)
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)

    with pytest.raises(TimeoutError):
        await c.acompletion(model="router-random-0.5",
                            messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_router_decision_weak_is_not_marked_downgraded(monkeypatch):
    """路由**判定**走弱 ≠ 降级。回归测试。

    实测背景：开启容错后，正常请求（win_rate < threshold 走弱模型）被误标
    `x-routellm-downgraded: true`。根因是 ResilientCaller 在 strong_fn=None
    分支无从区分「路由决策走弱」与「强模型失败降级」，一律标 downgraded。
    控制器必须显式纠正：只有「强模型失败后落到弱/缓存」才算降级。
    """
    c = _make_controller()
    async def _force_weak(*a, **k):
        return c.live_model_pair().weak  # 路由判定走弱

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_weak)
    c.resilience_enabled = True

    async def fake_acompletion(**kwargs):
        return _FakeResp("weak-by-routing")

    monkeypatch.setattr("routellm.controller.acompletion", fake_acompletion)

    res = await c.acompletion(model="router-random-0.9",
                              messages=[{"role": "user", "content": "hi"}])
    assert res.text == "weak-by-routing"
    assert c.last_downgraded is False, (
        "路由判定走弱不是降级，不应标记 downgraded"
    )


@pytest.mark.asyncio
async def test_router_failure_degrades_to_weak(monkeypatch):
    """降级链①：路由器彻底故障时应走弱模型，而不是把异常抛给客户端。

    回归背景：`_get_routed_model_for_completion` 原有兜底是「再调一次
    route()」——若路由器服务整体不可达（如 BERT 6070 挂掉），第二次调用
    同样失败，异常直接冒泡，永远到不了下游降级链。
    实测在 43 上以 RemoteInferenceError 复现。
    """
    c = _make_controller()
    calls: list[str] = []

    class RemoteInferenceError(Exception):
        pass

    # 路由器实例：calculate 与 route 均失败
    class _DeadRouter:
        def calculate_strong_win_rate(self, prompt):
            raise RemoteInferenceError("cannot reach inference service")

        def route(self, prompt, threshold, pair):
            raise RemoteInferenceError("cannot reach inference service")

    c.routers["random"] = _DeadRouter()

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return _FakeResp("weak-answer")

    monkeypatch.setattr("routellm.controller.acompletion", fake_acompletion)
    c.resilience_enabled = True
    c.resilience_max_attempts = 1

    res = await c.acompletion(model="router-random-0.5",
                              messages=[{"role": "user", "content": "hi"}])
    assert res.text == "weak-answer", "路由器故障应降级到弱模型"
    assert c.last_downgraded is True
    # 只应调用弱模型（不浪费强模型调用）
    assert all("weak" in m for m in calls), calls


@pytest.mark.asyncio
async def test_router_failure_without_resilience_still_raises(monkeypatch):
    """未启用容错时，路由器故障仍按原行为抛出（向后兼容）。"""
    c = _make_controller()

    class RemoteInferenceError(Exception):
        pass

    class _DeadRouter:
        def calculate_strong_win_rate(self, prompt):
            raise RemoteInferenceError("dead")

        def route(self, prompt, threshold, pair):
            raise RemoteInferenceError("dead")

    c.routers["random"] = _DeadRouter()

    with pytest.raises(RemoteInferenceError):
        await c.acompletion(model="router-random-0.5",
                            messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_retry_not_applied_to_bad_request(monkeypatch):
    """参数错误不应重试，也不应触发降级到弱模型（重试无意义）。"""
    c = _make_controller()
    calls: list[str] = []

    class BadRequestError(Exception):
        pass

    async def bad(**kwargs):
        calls.append(kwargs["model"])
        raise BadRequestError("400 invalid param")

    monkeypatch.setattr("routellm.controller.acompletion", bad)
    async def _force_strong(*a, **k):
        return c.live_model_pair().strong

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", _force_strong)
    c.resilience_enabled = True

    with pytest.raises(BadRequestError):
        await c.acompletion(model="router-random-0.5",
                            messages=[{"role": "user", "content": "hi"}])
    # 只尝试了强模型一次（不可重试，且不应降级 —— 弱模型同样会参数错）
    assert len(calls) == 1, f"BadRequest 不应重试/降级，实际调用 {calls}"
