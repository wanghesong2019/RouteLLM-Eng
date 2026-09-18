"""Controller 异步路由接入测试（方案文档 4.4 步骤 4）。

改造点：`acompletion` 目前在协程里直接调用**同步**的
`_get_routed_model_for_completion` —— 同步 HTTP/计算阻塞事件循环。

改造后应满足：
1. 存在异步路由路径 `_get_routed_model_for_completion_async`
2. `acompletion` 走异步路由（并发时不被路由阶段串行化）
3. 同步 `completion` / `route` 行为不变（向后兼容，评测脚本依赖）
4. 路由器无 `route_async` 时优雅回落到线程池包装（基类已提供）
"""

from __future__ import annotations

import asyncio
import time

import pytest

from routellm.controller import Controller


class _FakeResp:
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    def model_dump(self):
        return {"choices": [{"message": {"content": self.text}}]}


def _make_controller():
    return Controller(
        routers=["random"],
        strong_model="openai/strong-model",
        weak_model="openai/weak-model",
        api_base="http://x/v1",
        api_key="k",
    )


def test_controller_has_async_routing_path():
    c = _make_controller()
    assert hasattr(c, "_get_routed_model_for_completion_async"), \
        "缺少异步路由路径"


@pytest.mark.asyncio
async def test_async_routing_does_not_block_event_loop(monkeypatch):
    """异步路由不得阻塞事件循环（核心判据）。

    用一个「同步 sleep」的假路由器：若 Controller 在协程里直接调同步版本，
    事件循环会被卡住；走异步路径则应能并发推进。
    """
    c = _make_controller()

    class _SlowRouter:
        def calculate_strong_win_rate(self, prompt):
            time.sleep(0.1)          # 同步阻塞
            return 0.9

        def route(self, prompt, threshold, pair):
            return pair.strong if self.calculate_strong_win_rate(prompt) >= threshold else pair.weak

    c.routers["random"] = _SlowRouter()

    ticks = {"n": 0}

    async def ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.01)

    t = asyncio.create_task(ticker())
    await c._get_routed_model_for_completion_async(
        [{"role": "user", "content": "hi"}], "random", 0.5
    )
    t.cancel()

    assert ticks["n"] >= 3, (
        f"事件循环被阻塞（仅推进 {ticks['n']} 次，同步 sleep 0.1s）"
    )


@pytest.mark.asyncio
async def test_async_routing_returns_same_decision(monkeypatch):
    """异步路径的路由结果必须与同步一致。"""
    c = _make_controller()

    class _FixedRouter:
        def __init__(self, wr):
            self.wr = wr

        def calculate_strong_win_rate(self, prompt):
            return self.wr

        def route(self, prompt, threshold, pair):
            return pair.strong if self.wr >= threshold else pair.weak

    for wr, threshold, expect in (
        (0.9, 0.5, "strong"),
        (0.1, 0.5, "weak"),
        (0.5, 0.5, "strong"),   # 边界：>= 走强
    ):
        c.routers["random"] = _FixedRouter(wr)
        got = await c._get_routed_model_for_completion_async(
            [{"role": "user", "content": "hi"}], "random", threshold
        )
        pair = c.live_model_pair()
        assert got == (pair.strong if expect == "strong" else pair.weak), (
            f"wr={wr} th={threshold} 期望 {expect}，实际 {got}"
        )


@pytest.mark.asyncio
async def test_acompletion_uses_async_routing(monkeypatch):
    """acompletion 必须走异步路由路径（而不是同步版本）。"""
    c = _make_controller()
    used = {"async": False, "sync": False}

    async def fake_async(messages, router, threshold):
        used["async"] = True
        return c.live_model_pair().weak

    def fake_sync(messages, router, threshold):
        used["sync"] = True
        return c.live_model_pair().weak

    monkeypatch.setattr(c, "_get_routed_model_for_completion_async", fake_async)
    monkeypatch.setattr(c, "_get_routed_model_for_completion", fake_sync)

    async def fake_acompletion(**kwargs):
        return _FakeResp()

    monkeypatch.setattr("routellm.controller.acompletion", fake_acompletion)

    await c.acompletion(model="router-random-0.5",
                        messages=[{"role": "user", "content": "hi"}])
    assert used["async"] is True, "acompletion 未走异步路由路径"
    assert used["sync"] is False, "acompletion 仍在调用同步路由"


@pytest.mark.asyncio
async def test_async_routing_records_metrics():
    """异步路径同样要记录 RoutingInfo（监控不能因改造失效）。"""
    from routellm.controller import get_routing_info

    c = _make_controller()

    class _FixedRouter:
        def calculate_strong_win_rate(self, prompt):
            return 0.75

        def route(self, prompt, threshold, pair):
            return pair.strong

    c.routers["random"] = _FixedRouter()
    await c._get_routed_model_for_completion_async(
        [{"role": "user", "content": "hi"}], "random", 0.5
    )
    info = get_routing_info()
    assert info is not None, "异步路径未记录 RoutingInfo"
    assert info.router == "random"
    assert info.win_rate == pytest.approx(0.75)
    assert info.routed_model == "strong"


@pytest.mark.asyncio
async def test_router_failure_fallback_probe_does_not_write_contextvar():
    """回归：`_router_failure_fallback_probe` 必须**不写** ContextVar。

    背景：该方法在 asyncio.to_thread 的工作线程中执行，而线程里的
    ContextVar 写入不会传播回主协程 —— 曾导致「路由器故障降级」在
    异步路径下丢失降级标记（last_downgraded 应为 True 却为 False）。
    故约定：probe 只返回 (model, is_fallback)，标记由主协程设置。
    """
    from routellm.controller import is_router_fallback

    c = _make_controller()

    class _DeadRouter:
        def calculate_strong_win_rate(self, prompt):
            raise ConnectionError("dead")

        def route(self, prompt, threshold, pair):
            raise ConnectionError("dead")

    c.resilience_enabled = True
    pair = c.live_model_pair()

    # 主协程先清零，再在**线程**中调 probe
    import asyncio

    from routellm.controller import _set_router_fallback

    _set_router_fallback(False)

    model, is_fb = await asyncio.to_thread(
        c._router_failure_fallback_probe, _DeadRouter(), "hi", 0.5, pair
    )
    assert model == pair.weak
    assert is_fb is True
    # 线程内的 probe 不得改变主协程可见的标记
    assert is_router_fallback() is False, (
        "probe 在子线程写了 ContextVar —— 该写入不会传播回主协程，"
        "标记必须由调用方在主协程中设置"
    )


@pytest.mark.asyncio
async def test_async_router_failure_sets_downgrade_flag_end_to_end():
    """端到端：异步路径下路由器故障也必须置降级标记。"""
    c = _make_controller()
    c.resilience_enabled = True
    c.resilience_max_attempts = 1

    class _DeadRouter:
        def calculate_strong_win_rate(self, prompt):
            raise ConnectionError("dead")

        def route(self, prompt, threshold, pair):
            raise ConnectionError("dead")

    c.routers["random"] = _DeadRouter()

    async def fake_acompletion(**kwargs):
        return _FakeResp("weak-answer")

    import routellm.controller as ctrl
    ctrl_acompletion = ctrl.acompletion
    ctrl.acompletion = fake_acompletion
    try:
        res = await c.acompletion(model="router-random-0.5",
                                  messages=[{"role": "user", "content": "hi"}])
        assert res.text == "weak-answer"
        assert c.last_downgraded is True, "异步路径丢失了降级标记"
    finally:
        ctrl.acompletion = ctrl_acompletion


def test_sync_interface_unchanged():
    """同步路由接口保持可用（向后兼容）。"""
    c = _make_controller()

    class _FixedRouter:
        def calculate_strong_win_rate(self, prompt):
            return 0.9

        def route(self, prompt, threshold, pair):
            return pair.strong if 0.9 >= threshold else pair.weak

    c.routers["random"] = _FixedRouter()
    pair = c.live_model_pair()
    assert c.route("hi", "random", 0.5) == pair.strong
    assert c._get_routed_model_for_completion(
        [{"role": "user", "content": "hi"}], "random", 0.5
    ) == pair.strong
