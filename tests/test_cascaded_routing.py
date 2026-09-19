"""级联路由集成测试：FastPath + BERT + AdaptiveThreshold（方案文档步骤 4）。

用 deterministic 的假路由器替代 BERT（random 路由器返回随机值，无法断言
路由结果 —— 方案原稿里的断言依赖随机值，这里改为可控替身）。
"""

import pytest

from routellm.controller import Controller, get_routing_info
from routellm.routers.adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdController,
)
from routellm.routers.fast_path import FastPathRouter
from routellm.routers.routers import ROUTER_CLS


class _FixedRouter:
    """固定返回指定 win_rate 的假路由器。"""

    NO_PARALLEL = True

    def __init__(self, win_rate: float = 0.9):
        self.win_rate = win_rate
        self.calls = 0

    def calculate_strong_win_rate(self, prompt) -> float:
        self.calls += 1
        return self.win_rate

    def route(self, prompt, threshold, routed_pair):
        return (
            routed_pair.strong
            if self.calculate_strong_win_rate(prompt) >= threshold
            else routed_pair.weak
        )


@pytest.fixture
def fixed_router():
    return _FixedRouter(win_rate=0.9)


@pytest.fixture
def controller(fixed_router, monkeypatch):
    monkeypatch.setitem(ROUTER_CLS, "fixed", lambda **kw: fixed_router)
    return Controller(
        routers=["fixed"],
        strong_model="openai/strong",
        weak_model="openai/weak",
        api_base="http://x/v1",
        api_key="k",
        fast_path=FastPathRouter(),
        adaptive_threshold=AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_base=0.5), metrics_store=None
        ),
    )


@pytest.mark.asyncio
async def test_fast_path_bypasses_bert(controller, fixed_router):
    """简单 Query 应绕过 BERT，直接走弱模型。"""
    routed = await controller._get_routed_model_for_completion_async(
        [{"role": "user", "content": "你好"}], "fixed", 0.5
    )
    assert routed == controller.live_model_pair().weak
    assert fixed_router.calls == 0, "命中快速通道时不应调用 BERT"
    ri = get_routing_info()
    assert ri.router == "fast_path"
    assert ri.routed_model == "weak"


@pytest.mark.asyncio
async def test_complex_query_goes_to_bert(controller, fixed_router):
    """复杂 Query 应走真正的路由器（BERT）。"""
    routed = await controller._get_routed_model_for_completion_async(
        [{"role": "user", "content": "请详细分析量子计算中的纠错码原理"}],
        "fixed", 0.5,
    )
    assert fixed_router.calls == 1
    assert routed == controller.live_model_pair().strong
    ri = get_routing_info()
    assert ri.router == "fixed"


@pytest.mark.asyncio
async def test_adaptive_threshold_replaces_static(fixed_router, monkeypatch):
    """自适应阈值应替代静态阈值。"""
    monkeypatch.setitem(ROUTER_CLS, "fixed2", lambda **kw: fixed_router)
    c = Controller(
        routers=["fixed2"],
        strong_model="openai/strong",
        weak_model="openai/weak",
        api_base="http://x/v1",
        api_key="k",
        fast_path=FastPathRouter(),
        adaptive_threshold=AdaptiveThresholdController(
            AdaptiveThresholdConfig(tau_min=0.35, tau_max=0.75, tau_base=0.5),
            metrics_store=None,
        ),
    )
    # 自适应 τ 压到 0.35，静态阈值给 0.9；win_rate=0.4
    #   静态 0.9 → 弱；自适应 0.35 → 强
    c.adaptive_threshold._effective_tau = 0.35
    routed = await c._get_routed_model_for_completion_async(
        [{"role": "user", "content": "复杂问题需要详细分析至少二十个字符以上"}],
        "fixed2", 0.9,
    )
    assert routed == c.live_model_pair().strong


@pytest.mark.asyncio
async def test_backward_compatible_without_cascade(fixed_router, monkeypatch):
    """不注入 fast_path / adaptive_threshold 时行为与改造前一致。"""
    monkeypatch.setitem(ROUTER_CLS, "fixed3", lambda **kw: fixed_router)
    c = Controller(
        routers=["fixed3"],
        strong_model="openai/strong",
        weak_model="openai/weak",
        api_base="http://x/v1",
        api_key="k",
    )
    assert c.fast_path is None and c.adaptive_threshold is None
    routed = await c._get_routed_model_for_completion_async(
        [{"role": "user", "content": "你好"}], "fixed3", 0.5
    )
    # 无级联 → 照样走 BERT；win_rate=0.9 ≥ 0.5 → 强
    assert fixed_router.calls == 1
    assert routed == c.live_model_pair().strong
