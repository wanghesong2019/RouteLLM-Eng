"""异步化改造测试（方案文档 4.4）。

问题：FastAPI async 入口，但路由器全同步 —— 同步 HTTP 调用 + 同步 numpy/torch
计算阻塞事件循环。单 worker 下一个慢请求阻塞所有并发。

改造目标（文档 4.4）：
1. `Router` 基类增加 `route_async`，默认用 `asyncio.to_thread` 包装同步实现
2. `RemoteBERTRouter` 改用 `httpx.AsyncClient`（当前用同步 urllib）
3. `Controller` 走 `route_async`
4. 现有同步接口保持不变（向后兼容）

关键判据：在 async 上下文中，**并发请求不应串行化**。同步版会串行，
异步版应能重叠 —— 这是本文件的核心测试。
"""

from __future__ import annotations

import asyncio
import time

import pytest

# 注意：不能在模块顶层 import remote（与 routers.routers 存在导入顺序依赖，
# 顶层导入会触发 circular import）。改为在各测试内导入。
def _remote():
    from routellm.routers.remote import RemoteBERTRouter  # noqa: PLC0415
    return RemoteBERTRouter


def _err():
    from routellm.routers.remote import RemoteInferenceError  # noqa: PLC0415
    return RemoteInferenceError


class _FakeAsyncScorer:
    """替身：模拟推理服务的异步评分（可注入延迟）。"""

    def __init__(self, delay: float = 0.0, win_rate: float = 0.6):
        self.delay = delay
        self.win_rate = win_rate
        self.calls = 0
        self.concurrent_peak = 0
        self._inflight = 0

    async def score(self, prompts):
        self.calls += 1
        self._inflight += 1
        self.concurrent_peak = max(self.concurrent_peak, self._inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return [self.win_rate] * len(prompts)
        finally:
            self._inflight -= 1


# ---------------------------------------------------------------------------
# Router 基类契约
# ---------------------------------------------------------------------------

def test_router_base_has_route_async():
    """Router 基类必须提供 route_async（默认线程池包装同步实现）。"""
    from routellm.routers.routers import Router

    assert hasattr(Router, "route_async"), "Router 缺少 route_async"


@pytest.mark.asyncio
async def test_base_route_async_wraps_sync_via_thread():
    """基类默认实现：不应阻塞事件循环（须走线程池）。"""
    from routellm.routers.routers import Router

    ran_in_thread = {"v": False}

    class _SleepyRouter(Router):
        def calculate_strong_win_rate(self, prompt):
            import threading

            ran_in_thread["v"] = threading.current_thread() is not threading.main_thread()
            time.sleep(0.05)
            return 0.7

    r = _SleepyRouter()
    # 同步实现虽慢，但不应阻塞事件循环 —— 用「同时推进的计数器」验证
    ticks = {"n": 0}

    async def ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    t = asyncio.create_task(ticker())
    wr = await r.route_async("hi", 0.5, None)
    t.cancel()

    assert wr >= 0.5
    assert ran_in_thread["v"] is True, "同步实现应在工作线程中执行"
    assert ticks["n"] > 3, f"事件循环被阻塞了（仅推进 {ticks['n']} 次）"


# ---------------------------------------------------------------------------
# RemoteBERTRouter 异步接口
# ---------------------------------------------------------------------------

def test_remote_router_has_async_method():
    """RemoteBERTRouter 必须提供异步评分接口。"""
    r = _remote()(base_url="http://127.0.0.1:6070")
    assert hasattr(r, "calculate_strong_win_rate_async"), \
        "缺少 calculate_strong_win_rate_async"
    assert hasattr(r, "route_async"), "缺少 route_async"


@pytest.mark.asyncio
async def test_async_batch_preserves_order_and_values():
    """异步批量评分须保持顺序且值正确。"""
    r = _remote()(base_url="http://127.0.0.1:6070")
    r._async_score_prompts = lambda prompts: _return_sequence(prompts)
    out = await r.calculate_strong_win_rate_batch_async(["a", "b", "c"])
    assert out == [0.1, 0.2, 0.3]


async def _return_sequence(prompts):
    return [round(0.1 * (i + 1), 4) for i in range(len(prompts))]


@pytest.mark.asyncio
async def test_async_batch_chunks_by_batch_size():
    """异步批量评分须按 batch_size 分片，且保持顺序。"""
    r = _remote()(base_url="http://127.0.0.1:6070", batch_size=2)
    seen = []

    async def fake_score(prompts):
        seen.append(list(prompts))
        return [0.5] * len(prompts)

    r._async_score_prompts = fake_score
    out = await r.calculate_strong_win_rate_batch_async(["a", "b", "c", "d", "e"])
    assert out == [0.5] * 5
    assert seen == [["a", "b"], ["c", "d"], ["e"]], seen


@pytest.mark.asyncio
async def test_async_error_wrapped_as_remote_inference_error():
    """异步路径的失败也要包装成 RemoteInferenceError（与同步一致）。

    注意替换点：必须替换**底层 HTTP 客户端**，而不是
    `_async_score_prompts` —— 异常包装逻辑就在后者内部，
    把它整个替换掉等于绕过被测代码。
    """
    r = _remote()(base_url="http://127.0.0.1:6070")

    class _BoomClient:
        async def post(self, url, json=None):
            raise ConnectionError("refused")

        async def aclose(self):
            pass

    async def fake_get_client():
        return _BoomClient()

    r._get_async_client = fake_get_client
    with pytest.raises(_err()):
        await r.calculate_strong_win_rate_batch_async(["a"])


@pytest.mark.asyncio
async def test_async_http_error_status_wrapped():
    """HTTP >= 400 也应包装成 RemoteInferenceError，而不是抛出 httpx 异常。"""
    r = _remote()(base_url="http://127.0.0.1:6070")

    class _Resp:
        status_code = 500
        text = "internal error"

    class _BadClient:
        async def post(self, url, json=None):
            return _Resp()

        async def aclose(self):
            pass

    async def fake_get_client():
        return _BadClient()

    r._get_async_client = fake_get_client
    with pytest.raises(_err()) as e:
        await r.calculate_strong_win_rate_batch_async(["a"])
    assert "500" in str(e.value)


@pytest.mark.asyncio
async def test_async_empty_input_returns_empty():
    r = _remote()(base_url="http://127.0.0.1:6070")
    assert await r.calculate_strong_win_rate_batch_async([]) == []


# ---------------------------------------------------------------------------
# 并发性：异步版必须能重叠，同步版会串行
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_scoring_overlaps_concurrently():
    """核心判据：并发 10 个请求时，异步版耗时应远小于串行。

    同步阻塞版（urllib）在 async 上下文中会串行执行 —— 单 worker 下
    一个慢请求阻塞所有并发，这正是问题4。
    """
    delay = 0.1
    r = _remote()(base_url="http://127.0.0.1:6070")
    scorer = _FakeAsyncScorer(delay=delay)

    async def fake_score(prompts):
        return await scorer.score(prompts)

    r._async_score_prompts = fake_score

    t0 = time.perf_counter()
    await asyncio.gather(*[r.calculate_strong_win_rate_async("q") for _ in range(10)])
    elapsed = time.perf_counter() - t0

    # 串行需 >= 10*0.1 = 1.0s；并发应明显更短
    assert elapsed < 0.5, (
        f"并发未生效：10 请求 × {delay}s 用了 {elapsed:.2f}s（串行约 1.0s）"
    )
    assert scorer.concurrent_peak > 1, (
        f"未观察到并发（峰值 {scorer.concurrent_peak}）"
    )


@pytest.mark.asyncio
async def test_async_uses_connection_pooling_client():
    """应复用 httpx.AsyncClient（连接池），而非每次新建 —— 否则并发退化为串行。"""
    r = _remote()(base_url="http://127.0.0.1:6070")
    client1 = await r._get_async_client()
    client2 = await r._get_async_client()
    assert client1 is client2, "AsyncClient 未被复用（连接池失效）"
    await r.aclose()


# ---------------------------------------------------------------------------
# 向后兼容
# ---------------------------------------------------------------------------

def test_sync_interface_unchanged():
    """同步接口必须保持可用（向后兼容，评测脚本依赖）。"""
    r = _remote()(base_url="http://127.0.0.1:6070")
    assert callable(r.calculate_strong_win_rate)
    assert callable(r.calculate_strong_win_rate_batch)

    calls = []

    def fake_post(payload):
        calls.append(payload)
        return {"results": [{"win_rate": 0.7} for _ in payload["prompts"]]}

    r._post_score = fake_post
    assert r.calculate_strong_win_rate("x") == 0.7
    assert r.calculate_strong_win_rate_batch(["x", "y"]) == [0.7, 0.7]
