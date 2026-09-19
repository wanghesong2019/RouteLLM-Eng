"""窗口计数器重启恢复测试。

问题：WindowMetricsCounter 是纯内存累加，容器重启即归零 → 重启后 cost_rate=0
→ τ 回落 τ_base → 关闭期间/重启前的真实消耗被"遗忘"，闭环短暂失效。

修法：启动时从 MetricsStore 回填窗口内的历史样本。
"""

import time

import pytest

from routellm.monitoring.metrics import RequestMetrics
from routellm.monitoring.store import MetricsStore
from routellm.routers.adaptive_threshold import WindowMetricsCounter


def _metric(ts: float, i: int, pt: int = 100, ct: int = 50, lat: float = 200.0):
    return RequestMetrics(
        request_id=f"r{i}",
        timestamp=ts,
        prompt_hash="h",
        router_name="remote_bert",
        threshold=0.5,
        win_rate=0.6,
        routed_model="weak",
        routing_latency_ms=10.0,
        llm_latency_ms=190.0,
        total_latency_ms=lat,
        prompt_tokens=pt,
        completion_tokens=ct,
        estimated_cost=0.001,
        cache_hit=False,
        status="success",
    )


@pytest.mark.asyncio
async def test_warmup_from_store_restores_window_samples():
    """重启后 warmup 应把窗口内的历史 token 数捞回来。"""
    store = MetricsStore(":memory:")
    now = time.time()
    for i in range(10):
        await store.save(_metric(now - 20, i, pt=100, ct=100))  # 每次 200 tok

    counter = WindowMetricsCounter(window_sec=60.0)
    # 重启后：内存为空 → 速率 0（这就是要修的故障）
    before = counter.snapshot(now - 60)
    assert before["tokens_per_min"] == 0.0
    assert before["total_count"] == 0

    await counter.warmup(store)

    after = counter.snapshot(now - 60)
    assert after["total_count"] == 10, "未从 store 回填窗口内样本"
    assert after["tokens_per_min"] > 0


@pytest.mark.asyncio
async def test_warmup_excludes_out_of_window():
    """窗口外的历史不应被回填（否则速率会被历史拉高）。"""
    store = MetricsStore(":memory:")
    now = time.time()
    await store.save(_metric(now - 30, 0, pt=100, ct=100))    # 窗口内
    await store.save(_metric(now - 300, 1, pt=9999, ct=9999))  # 窗口外

    counter = WindowMetricsCounter(window_sec=60.0)
    await counter.warmup(store)
    snap = counter.snapshot(now - 60)
    assert snap["total_count"] == 1, "窗口外样本不应被回填"
    # 只有 200 tok（不是 200+19998）
    assert snap["tokens_per_min"] < 1000


@pytest.mark.asyncio
async def test_warmup_restores_latency_p90_histogram():
    """重启后 P90 延迟也应恢复（分桶直方图需一并回填）。"""
    store = MetricsStore(":memory:")
    now = time.time()
    for i in range(10):
        await store.save(_metric(now - 10, i, lat=500.0 + i * 100))

    counter = WindowMetricsCounter(window_sec=60.0)
    assert counter.snapshot(now - 60)["latency_p90"] == 0.0

    await counter.warmup(store)
    p90 = counter.snapshot(now - 60)["latency_p90"]
    assert p90 > 0, "重启后 P90 延迟未恢复"


@pytest.mark.asyncio
async def test_warmup_is_failure_tolerant():
    """store 不可用时 warmup 不应抛异常（计数器退回冷启动即可）。"""

    class BrokenStore:
        async def window_samples(self, since):
            raise RuntimeError("db down")

    counter = WindowMetricsCounter(window_sec=60.0)
    # 不抛异常
    await counter.warmup(BrokenStore())
    assert counter.snapshot(time.time() - 60)["total_count"] == 0


@pytest.mark.asyncio
async def test_warmup_then_record_accumulates():
    """warmup 后新样本应继续累加，不是覆盖。"""
    store = MetricsStore(":memory:")
    now = time.time()
    await store.save(_metric(now - 10, 0, pt=100, ct=100))

    counter = WindowMetricsCounter(window_sec=60.0)
    await counter.warmup(store)
    counter.record(50)  # 新请求

    snap = counter.snapshot(now - 60)
    assert snap["total_count"] == 2
