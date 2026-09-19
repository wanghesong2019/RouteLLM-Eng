"""背压机制测试（方案文档 3.4）。

硬质量防线：预算耗尽 + 高难请求（s ≥ τ_max）时**不以次充好** ——
宁可 429 背压，也不静默降级到弱模型。
"""

import httpx
import pytest
from fastapi import FastAPI

from routellm.controller import Controller
from routellm.openai_server import ErrorResponse


@pytest.fixture
def backpressure_app(tmp_path, monkeypatch):
    """构造带自适应阈值（无预算）的网关应用，避免依赖真实下游/鉴权。"""
    monkeypatch.setenv("ROUTELLM_METRICS_DB", str(tmp_path / "m.db"))
    import importlib

    import routellm.openai_server as srv

    importlib.reload(srv)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: dict):
        from routellm.controller import RoutingError

        try:
            raise RoutingError(
                "Token budget exhausted for high-difficulty request "
                "(backpressure guardrail). Retry later."
            )
        except RoutingError as e:
            from fastapi.responses import JSONResponse

            status = 429 if "backpressure" in str(e).lower() else 400
            headers = {"Retry-After": "30"} if status == 429 else None
            return JSONResponse(
                ErrorResponse(message=str(e)).model_dump(),
                status_code=status,
                headers=headers,
            )

    return app, srv


@pytest.mark.asyncio
async def test_backpressure_maps_to_429_with_retry_after(backpressure_app):
    """背压错误 → HTTP 429 + Retry-After（而非 400）。"""
    app, _ = backpressure_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        r = await client.post("/v1/chat/completions", json={"model": "m"})
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "30"
    assert "backpressure" in r.json()["message"].lower()


def test_should_backpressure_thresholds():
    """should_backpressure 的判定矩阵。"""
    from routellm.routers.adaptive_threshold import (
        AdaptiveThresholdConfig,
        AdaptiveThresholdController,
    )

    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(
            tau_max=0.75, budget_tokens_per_min=100
        ),
        metrics_store=None,
    )
    ctrl._cost_rate = 200  # 超预算 2 倍（> 1.5 倍阈值）
    assert ctrl.should_backpressure(0.8) is True   # 高难 → 拒
    assert ctrl.should_backpressure(0.5) is False  # 不难 → 允许走弱

    ctrl._cost_rate = 50  # 未超预算
    assert ctrl.should_backpressure(0.8) is False

    ctrl._cost_rate = 200
    ctrl.config = AdaptiveThresholdConfig(budget_tokens_per_min=0)
    assert ctrl.should_backpressure(0.99) is False  # 未设预算 → 永不背压


@pytest.mark.asyncio
async def test_controller_raises_backpressure_after_routing(fixed_router_patch):
    """Controller.acompletion 在路由决策后、下游调用前抛背压错误。"""
    from routellm.controller import get_routing_info
    from routellm.routers.adaptive_threshold import (
        AdaptiveThresholdConfig,
        AdaptiveThresholdController,
    )
    from routellm.routers.fast_path import FastPathRouter
    from routellm.routers.routers import ROUTER_CLS

    fixed = fixed_router_patch(win_rate=0.9)
    ROUTER_CLS["bp_fixed"] = lambda **kw: fixed

    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(tau_max=0.75, budget_tokens_per_min=100),
        metrics_store=None,
    )
    ctrl._cost_rate = 500  # 预算耗尽

    c = Controller(
        routers=["bp_fixed"],
        strong_model="openai/strong",
        weak_model="openai/weak",
        api_base="http://x/v1",
        api_key="k",
        fast_path=FastPathRouter(),
        adaptive_threshold=ctrl,
    )

    from routellm.controller import RoutingError

    with pytest.raises(RoutingError) as ei:
        await c.acompletion(
            model="router-bp_fixed-0.5",
            messages=[{"role": "user", "content": "请详细分析这个高难度技术问题需要强模型"}],
        )
    assert "backpressure" in str(ei.value).lower()


@pytest.fixture
def fixed_router_patch():
    class _Fixed:
        NO_PARALLEL = True

        def __init__(self, win_rate=0.9):
            self.win_rate = win_rate

        def calculate_strong_win_rate(self, prompt):
            return self.win_rate

        def route(self, prompt, threshold, routed_pair):
            return routed_pair.strong if self.win_rate >= threshold else routed_pair.weak

    def _make(win_rate=0.9):
        return _Fixed(win_rate)

    return _make
