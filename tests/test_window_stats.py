"""MetricsStore 窗口查询测试（方案文档 3.3）。

自适应阈值控制器需要「滑动窗口内」的成本/延迟聚合，而非全量 summary。
"""

import time

import pytest

from routellm.monitoring.metrics import RequestMetrics
from routellm.monitoring.store import MetricsStore, _percentile


def _metric(ts: float, i: int = 0, routed: str = "strong",
            pt: int = 100, ct: int = 50, lat: float = 215.0) -> RequestMetrics:
    return RequestMetrics(
        request_id=f"r{i}",
        timestamp=ts,
        prompt_hash="hash",
        router_name="remote_bert",
        threshold=0.5,
        win_rate=0.6,
        routed_model=routed,
        routing_latency_ms=15.0,
        llm_latency_ms=200.0,
        total_latency_ms=lat,
        prompt_tokens=pt,
        completion_tokens=ct,
        estimated_cost=0.001,
        cache_hit=False,
        status="success",
    )


@pytest.mark.asyncio
async def test_window_stats_empty():
    """空库应返回全零。"""
    store = MetricsStore(":memory:")
    stats = await store.window_stats(time.time() - 60)
    assert stats["tokens_per_min"] == 0.0
    assert stats["latency_p90"] == 0.0
    assert stats["total_count"] == 0
    assert stats["strong_count"] == 0
    assert stats["weak_count"] == 0
    assert stats["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_window_stats_with_data():
    """有数据时应正确聚合。"""
    store = MetricsStore(":memory:")
    now = time.time()
    for i in range(10):
        await store.save(
            _metric(now - 30, i, routed="strong" if i % 2 == 0 else "weak")
        )

    stats = await store.window_stats(now - 60)
    assert stats["total_count"] == 10
    assert stats["strong_count"] == 5
    assert stats["weak_count"] == 5
    assert stats["tokens_per_min"] > 0
    assert stats["latency_p90"] > 0


@pytest.mark.asyncio
async def test_window_stats_excludes_old_data():
    """窗口外的数据不应被统计。"""
    store = MetricsStore(":memory:")
    now = time.time()
    await store.save(_metric(now - 120, 0))  # 60s 窗口之外

    stats = await store.window_stats(now - 60)
    assert stats["total_count"] == 0


def test_percentile_helper():
    """_percentile 边界：空列表 → 0.0，单元素 → 自身，否则取位。

    取位约定与既有 _percentiles 一致：idx = round(q × (n-1))
    （0..9 的 P90 → idx 8 → 8.0）。
    """
    assert _percentile([], 0.9) == 0.0
    assert _percentile([5.0], 0.9) == 5.0
    vals = [float(i) for i in range(10)]  # 0..9
    assert _percentile(vals, 0.5) == 4.0
    assert _percentile(vals, 0.9) == 8.0
    assert _percentile(vals, 1.0) == 9.0
