"""容错与 Resilience 测试（方案文档 4.3）。

对应「问题3：无容错机制——单点故障即全盘崩溃」：
改造前 `Controller.acompletion()` 裸调 `litellm.acompletion()`，
无重试 / 超时 / 熔断，任一上游 API 故障 = 服务不可用。

设计（文档 4.3 方案 B：熔断器 + 重试 + 降级）：

1. 三态熔断器 CLOSED → OPEN → HALF_OPEN → CLOSED
   - failure_threshold=5（连续失败 5 次开路）
   - recovery_timeout=60（60s 后半开）
   - half_open_max_calls=3（半开态最多放行 3 次探测）

2. 指数退避重试 max_attempts=3, base_delay=1s, max_delay=10s, 带抖动
   - 只重试可恢复异常（TimeoutError / RateLimitError）
   - 不重试 BadRequestError（参数错误，重试无意义）

3. 降级链
   ① 路由器异常 → 直接走弱模型
   ② 强模型失败 → 降级到弱模型
   ③ 弱模型也失败 → 查缓存返回历史响应（header X-RouteLLM-Downgraded: true）
   ④ 缓存也无 → 503 + Retry-After: 30

TDD：本文件先写测试（RED），确认失败后再实现（GREEN）。
"""

from __future__ import annotations

import asyncio

import pytest

from routellm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    FallbackExhaustedError,
    ResilientCaller,
    is_retryable,
    with_retry,
)


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------

class _FakeClock:
    """可控时钟，避免测试真的 sleep。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request() is True


def test_circuit_breaker_opens_after_threshold_failures():
    """连续失败达到阈值 → 开路。

    返回的 bool（而非抛异常）符合 CircuitBreaker 的职责边界：
    它只回答「现在能不能放行」，调用方决定如何降级。
    """
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    for _ in range(2):
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow_request() is False


def test_circuit_breaker_success_resets_failure_count():
    """成功一次应清零连续失败计数（不是累计失败）。"""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # 若累计计数会开路；连续计数应仍为 CLOSED
    assert cb.state is CircuitState.CLOSED


def test_circuit_breaker_half_open_after_recovery_timeout():
    """开路后经过恢复期 → 半开，放行探测。"""
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60, clock=clock)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow_request() is False

    clock.advance(59)
    assert cb.allow_request() is False, "恢复期未到不应放行"

    clock.advance(2)
    assert cb.allow_request() is True
    assert cb.state is CircuitState.HALF_OPEN


def test_circuit_breaker_half_open_limits_concurrent_probes():
    """半开态最多放行 half_open_max_calls 次探测。"""
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60,
                        half_open_max_calls=3, clock=clock)
    cb.record_failure()
    cb.record_failure()
    clock.advance(61)

    allowed = sum(1 for _ in range(3) if cb.allow_request())
    assert allowed == 3
    assert cb.allow_request() is False, "超过 half_open_max_calls 应拒绝"


def test_circuit_breaker_half_open_success_closes():
    """半开态下探测成功 → 回到闭合。"""
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60, clock=clock)
    cb.record_failure()
    cb.record_failure()
    clock.advance(61)
    assert cb.allow_request() is True
    cb.record_success()
    assert cb.state is CircuitState.CLOSED
    assert cb.allow_request() is True


def test_circuit_breaker_half_open_failure_reopens():
    """半开态下探测失败 → 重新开路。"""
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60, clock=clock)
    cb.record_failure()
    cb.record_failure()
    clock.advance(61)
    cb.allow_request()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_circuit_breaker_rejects_invalid_params():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)


# ---------------------------------------------------------------------------
# 重试
# ---------------------------------------------------------------------------

def test_is_retryable_classification():
    """只重试可恢复异常；参数错误不重试。"""
    assert is_retryable(TimeoutError("t")) is True
    assert is_retryable(asyncio.TimeoutError()) is True
    assert is_retryable(ConnectionError("c")) is True

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    assert is_retryable(RateLimitError("429")) is True
    assert is_retryable(BadRequestError("400")) is False
    # 未知异常默认不重试（保守，避免放大故障）
    assert is_retryable(ValueError("x")) is False


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_third_attempt():
    """前两次失败、第三次成功 → 返回成功结果。"""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("boom")
        return "ok"

    res = await with_retry(flaky, max_attempts=3, base_delay=0.0,
                           max_delay=0.0, jitter=0.0)
    assert res == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_retry_gives_up_after_max_attempts():
    """超过 max_attempts 仍失败 → 抛出最后一次异常。"""

    async def always_fail():
        raise TimeoutError("always")

    with pytest.raises(TimeoutError):
        await with_retry(always_fail, max_attempts=3, base_delay=0.0,
                         max_delay=0.0, jitter=0.0)


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_non_retryable():
    """不可重试异常应立即抛出，只调用一次。"""
    calls = {"n": 0}

    class BadRequestError(Exception):
        pass

    async def bad():
        calls["n"] += 1
        raise BadRequestError("400")

    with pytest.raises(BadRequestError):
        await with_retry(bad, max_attempts=3, base_delay=0.0, max_delay=0.0,
                         jitter=0.0)
    assert calls["n"] == 1, "不可重试异常不应被重试"


@pytest.mark.asyncio
async def test_with_retry_backoff_is_exponential(monkeypatch):
    """退避延迟应按 base * 2^n 增长，且不超过 max_delay。"""
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fail():
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        await with_retry(always_fail, max_attempts=4, base_delay=1.0,
                         max_delay=10.0, jitter=0.0)

    # 3 次重试间隔：1, 2, 4
    assert sleeps == [1.0, 2.0, 4.0], sleeps


@pytest.mark.asyncio
async def test_with_retry_backoff_capped_at_max_delay(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fail():
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        await with_retry(always_fail, max_attempts=5, base_delay=1.0,
                         max_delay=3.0, jitter=0.0)

    assert sleeps == [1.0, 2.0, 3.0, 3.0], sleeps


# ---------------------------------------------------------------------------
# 降级链（ResilientCaller）
# ---------------------------------------------------------------------------

class _FakeCache:
    """极简缓存替身，接口对齐 MultiTierCache。"""

    def __init__(self, store: dict | None = None) -> None:
        self.store = dict(store or {})
        self.gets: list[str] = []

    def get(self, key):
        self.gets.append(key)
        return self.store.get(key)

    def set(self, key, value, ttl=None):
        self.store[key] = value


@pytest.mark.asyncio
async def test_fallback_strong_fails_goes_weak():
    """降级链②：强模型失败 → 降级到弱模型，并标记 downgraded。"""
    cache = _FakeCache()
    caller = ResilientCaller(cache=cache)

    async def call_strong():
        raise TimeoutError("strong down")

    async def call_weak():
        return "weak-answer"

    res = await caller.call_with_fallback(
        strong_fn=call_strong,
        weak_fn=call_weak,
        cache_key="k1",
        max_attempts=1,
    )
    assert res.value == "weak-answer"
    assert res.downgraded is True
    assert res.tier == "weak"
    assert res.source == "weak"


@pytest.mark.asyncio
async def test_fallback_both_fail_uses_cache():
    """降级链③：强弱都失败 → 查缓存返回历史响应。"""
    cache = _FakeCache({"k1": "cached-answer"})
    caller = ResilientCaller(cache=cache)

    async def fail():
        raise TimeoutError("down")

    res = await caller.call_with_fallback(
        strong_fn=fail, weak_fn=fail, cache_key="k1", max_attempts=1
    )
    assert res.value == "cached-answer"
    assert res.source == "cache"
    assert res.downgraded is True
    assert "k1" in cache.gets


@pytest.mark.asyncio
async def test_fallback_all_exhausted_raises_503_info():
    """降级链④：缓存也无 → 抛 FallbackExhaustedError（供上层转 503）。"""
    cache = _FakeCache()
    caller = ResilientCaller(cache=cache)

    async def fail():
        raise TimeoutError("down")

    with pytest.raises(FallbackExhaustedError) as e:
        await caller.call_with_fallback(
            strong_fn=fail, weak_fn=fail, cache_key="missing", max_attempts=1
        )
    assert e.value.retry_after == 30


@pytest.mark.asyncio
async def test_fallback_success_is_not_marked_downgraded():
    """强模型成功 → 不应标记 downgraded。"""
    cache = _FakeCache()
    caller = ResilientCaller(cache=cache)

    async def ok():
        return "strong-answer"

    async def weak():
        return "weak-answer"

    res = await caller.call_with_fallback(
        strong_fn=ok, weak_fn=weak, cache_key="k", max_attempts=1
    )
    assert res.value == "strong-answer"
    assert res.downgraded is False
    assert res.tier == "strong"


@pytest.mark.asyncio
async def test_fallback_circuit_open_skips_strong_entirely():
    """熔断开路时不应再尝试强模型（避免无谓等待）。"""
    cache = _FakeCache()
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    caller = ResilientCaller(cache=cache, strong_breaker=cb)
    cb.record_failure()  # 开路
    assert cb.state is CircuitState.OPEN

    calls = {"strong": 0, "weak": 0}

    async def strong():
        calls["strong"] += 1
        return "should-not-happen"

    async def weak():
        calls["weak"] += 1
        return "weak-answer"

    res = await caller.call_with_fallback(
        strong_fn=strong, weak_fn=weak, cache_key="k", max_attempts=1
    )
    assert calls["strong"] == 0, "开路时不应调用强模型"
    assert calls["weak"] == 1
    assert res.downgraded is True


@pytest.mark.asyncio
async def test_fallback_router_error_goes_weak():
    """降级链①：路由器异常 → 直接走弱模型。"""
    cache = _FakeCache()
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    caller = ResilientCaller(cache=cache, router_breaker=cb)
    cb.record_failure()  # 模拟路由器已开路

    calls = {"weak": 0}

    async def weak():
        calls["weak"] += 1
        return "weak-answer"

    res = await caller.call_with_fallback(
        strong_fn=None, weak_fn=weak, cache_key="k", max_attempts=1,
        router_ok=False,
    )
    assert res.value == "weak-answer"
    assert res.tier == "weak"
    assert res.downgraded is True


@pytest.mark.asyncio
async def test_fallback_client_error_not_swallowed_into_503():
    """客户端错误必须透传，不得降级成 503（回归测试）。

    否则一个本该 400 的参数错误会走完降级链、最终返回
    FallbackExhaustedError(503)，误导客户端去重试一个永远不会成功的请求。
    """
    cache = _FakeCache()
    caller = ResilientCaller(cache=cache)

    class BadRequestError(Exception):
        pass

    async def strong():
        raise BadRequestError("400 invalid param")

    async def weak():
        raise AssertionError("弱模型不该被调用 —— 参数错误换模型无法修复")

    with pytest.raises(BadRequestError):
        await caller.call_with_fallback(
            strong_fn=strong, weak_fn=weak, cache_key="k", max_attempts=1
        )

    # 路由器判定走弱时同理：弱模型的客户端错误也应透传
    async def weak_bad():
        raise BadRequestError("400")

    with pytest.raises(BadRequestError):
        await caller.call_with_fallback(
            strong_fn=None, weak_fn=weak_bad, cache_key="k", max_attempts=1
        )


@pytest.mark.asyncio
async def test_fallback_cache_key_format_matches_cache_module():
    """缓存兜底 key 应复用 cache.keys.result_key 的格式（routellm:res:...）。"""
    from routellm.cache.keys import result_key

    key = result_key("hello")
    assert key.startswith("routellm:res:")
    cache = _FakeCache({key: "cached"})
    caller = ResilientCaller(cache=cache)

    async def fail():
        raise TimeoutError("down")

    res = await caller.call_with_fallback(
        strong_fn=fail, weak_fn=fail, cache_key=key, max_attempts=1
    )
    assert res.value == "cached"
