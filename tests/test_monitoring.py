"""监控体系与 Dashboard 的单元测试（TDD RED 阶段）。

设计依据：方案文档 4.2 节（P0）—— 方案 B + A 混合
    自研 Dashboard（FastAPI + SQLite + 单页 HTML/ECharts）
    + Prometheus metrics 端点（兼容标准生态）

问题定位（文档 4.2.1）：
    改造前 openai_server.py 只有 `model_counts` 全局 defaultdict（重启丢失）
    + `logging.info`（无结构化），/health 只返回 {"status":"online"}。
    缺失：路由决策、win_rate 置信度、延迟分位、成本节省、缓存命中率。

指标体系（文档 4.2.4）：
    请求级：router_name / threshold / win_rate / routed_model /
            routing_latency_ms / llm_latency_ms / total_latency_ms /
            prompt_tokens / completion_tokens / estimated_cost / cache_hit / status

运行：
    pytest tests/test_monitoring.py -v
"""

import asyncio
import os

import pytest


@pytest.fixture
def store(tmp_path):
    """临时 SQLite store。"""
    from routellm.monitoring.store import MetricsStore

    return MetricsStore(str(tmp_path / "metrics.db"))


def _sample(store, **overrides):
    """构造并保存一条样本指标。"""
    from routellm.monitoring.metrics import RequestMetrics

    defaults = dict(
        request_id="req-1",
        timestamp=1000.0,
        prompt_hash="abc123",
        router_name="remote_bert",
        threshold=0.5,
        win_rate=0.7,
        routed_model="strong",
        routing_latency_ms=6.1,
        llm_latency_ms=1200.0,
        total_latency_ms=1210.0,
        prompt_tokens=10,
        completion_tokens=50,
        estimated_cost=0.001,
        cache_hit=False,
        status="success",
    )
    defaults.update(overrides)
    m = RequestMetrics(**defaults)
    asyncio.run(store.save(m))
    return m


# ------------------------------------------------------------ 数据模型


def test_metrics_model_fields():
    """指标模型须含文档定义的字段。"""
    from routellm.monitoring.metrics import RequestMetrics

    required = {
        "request_id", "timestamp", "prompt_hash", "router_name", "threshold",
        "win_rate", "routed_model", "routing_latency_ms", "llm_latency_ms",
        "total_latency_ms", "prompt_tokens", "completion_tokens",
        "estimated_cost", "cache_hit", "status",
    }
    fields = set(RequestMetrics.__dataclass_fields__)
    missing = required - fields
    assert not missing, f"缺少字段: {missing}"


def test_prompt_hash_not_plaintext():
    """指标不得存储 prompt 明文（隐私）。"""
    from routellm.monitoring.metrics import RequestMetrics

    assert "prompt" not in RequestMetrics.__dataclass_fields__, (
        "不应直接存 prompt 字段（只存 hash）"
    )


def test_cost_calculation_helper():
    """成本估算辅助函数应可算出合理值。"""
    from routellm.monitoring.metrics import estimate_cost

    # 强模型贵、弱模型便宜
    strong = estimate_cost("strong", 1000, 1000)
    weak = estimate_cost("weak", 1000, 1000)
    assert strong > weak > 0
    # 0 token → 0 成本
    assert estimate_cost("strong", 0, 0) == 0.0


# ------------------------------------------------------------ 存储


def test_store_save_and_count(store):
    """保存后应能统计总数。"""
    _sample(store, request_id="r1")
    _sample(store, request_id="r2")
    s = asyncio.run(store.summary())
    assert s["total_requests"] == 2


def test_store_latency_percentiles(store):
    """应能算出延迟分位数 P50/P95/P99。"""
    for i, lat in enumerate([100, 110, 120, 130, 500]):
        _sample(store, request_id=f"r{i}", total_latency_ms=float(lat))

    s = asyncio.run(store.summary())
    lat = s["latency_ms"]
    for k in ("p50", "p95", "p99", "mean"):
        assert k in lat, f"缺少 {k}"
    assert 100 <= lat["p50"] <= 130
    assert lat["p99"] >= 400


def test_store_strong_ratios(store):
    """应统计强/弱模型占比。"""
    _sample(store, request_id="r1", routed_model="strong")
    _sample(store, request_id="r2", routed_model="strong")
    _sample(store, request_id="r3", routed_model="weak")

    s = asyncio.run(store.summary())
    assert s["strong_count"] == 2
    assert s["weak_count"] == 1
    assert s["strong_ratio"] == pytest.approx(2 / 3, abs=1e-3)


def test_store_cost_savings(store):
    """应能算出「相比全走强模型」的节省额。

    注意：成本由 store 按 token 与 routed_model 统一重算，不信任传入的
    estimated_cost —— 避免口径不一致导致节省额失真（实测踩过）。
    """
    _sample(store, request_id="r1", routed_model="weak",
            prompt_tokens=1000, completion_tokens=1000)
    _sample(store, request_id="r2", routed_model="strong",
            prompt_tokens=1000, completion_tokens=1000)

    s = asyncio.run(store.summary())
    # 两条都按 token 算：弱模型便宜、强模型贵
    assert s["actual_cost_usd"] > 0
    assert s["cost_if_strong_usd"] > s["actual_cost_usd"], (
        f"若全走强模型应更贵: {s}"
    )
    assert s["cost_saved_usd"] > 0, f"应算出节省: {s}"


def test_store_cache_hit_rate(store):
    """应统计缓存命中率。"""
    _sample(store, request_id="r1", cache_hit=True)
    _sample(store, request_id="r2", cache_hit=True)
    _sample(store, request_id="r3", cache_hit=False)

    s = asyncio.run(store.summary())
    assert s["cache_hit_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_store_error_rate(store):
    """应统计错误率。"""
    _sample(store, request_id="r1", status="success")
    _sample(store, request_id="r2", status="error")

    s = asyncio.run(store.summary())
    assert s["error_rate"] == pytest.approx(0.5, abs=1e-3)


def test_store_recent_requests(store):
    """应能取最近请求列表（供 Dashboard 表格）。"""
    for i in range(5):
        _sample(store, request_id=f"r{i}", timestamp=1000.0 + i)

    rows = asyncio.run(store.recent(limit=3))
    assert len(rows) == 3
    # 按时间倒序（最新的在前）
    assert rows[0]["request_id"] == "r4"


def test_store_time_series(store):
    """应能按时间桶聚合（供折线图）。"""
    for i in range(10):
        _sample(store, request_id=f"r{i}", timestamp=1000.0 + i * 10)

    series = asyncio.run(store.time_series(bucket_seconds=30))
    assert len(series) >= 2, f"应分成多个时间桶: {series}"
    assert "bucket" in series[0] and "count" in series[0]


def test_store_empty_returns_zeros(store):
    """空库时应返回零值而非报错。"""
    s = asyncio.run(store.summary())
    assert s["total_requests"] == 0
    assert s["strong_ratio"] == 0.0


# ------------------------------------------------------------ Prometheus 端点


def test_prometheus_text_format():
    """Prometheus 端点须输出文本格式指标。"""
    from routellm.monitoring import prometheus

    prometheus.reset()
    prometheus.inc_counter("routellm_requests_total", {"model": "strong"})
    text = prometheus.render_metrics()
    assert isinstance(text, str)
    assert "routellm_requests_total" in text
    # 空的时候也不应报错
    prometheus.reset()
    assert prometheus.render_metrics().strip() == ""


def test_prometheus_records_requests():
    """记录请求后计数器应增长。"""
    from routellm.monitoring import prometheus

    prometheus.reset()
    before = prometheus.get_counter("routellm_requests_total", {"model": "strong"})
    prometheus.inc_counter("routellm_requests_total", {"model": "strong"})
    after = prometheus.get_counter("routellm_requests_total", {"model": "strong"})
    assert after == before + 1

    text = prometheus.render_metrics()
    assert 'routellm_requests_total{model="strong"}' in text


def test_prometheus_histogram():
    """直方图应记录观测值分布。"""
    from routellm.monitoring import prometheus

    prometheus.reset()
    for v in [10, 50, 120, 300]:
        prometheus.observe_histogram("routellm_latency_ms", v)
    text = prometheus.render_metrics()
    assert "routellm_latency_ms_bucket" in text
    assert "routellm_latency_ms_count 4" in text


# ------------------------------------------------------------ Dashboard API


def test_dashboard_routes_registered():
    """Dashboard 相关路由须注册。"""
    from routellm.monitoring.dashboard import router

    paths = {r.path for r in router.routes}
    for p in ("/dashboard", "/api/metrics/summary", "/api/metrics/recent"):
        assert p in paths, f"缺少路由 {p}"


def test_dashboard_summary_api(store):
    """summary API 应返回前端所需结构。"""
    from routellm.monitoring.dashboard import api_summary

    _sample(store, request_id="r1", routed_model="strong")
    _sample(store, request_id="r2", routed_model="weak")

    # 注入 store
    from routellm.monitoring import dashboard

    dashboard.set_store(store)
    res = asyncio.run(api_summary())
    assert res["total_requests"] == 2
    assert "cost_saved_usd" in res
    assert "latency_ms" in res


def test_dashboard_static_html_exists():
    """Dashboard 页面文件须存在（单页 HTML）。"""
    from routellm.monitoring.dashboard import STATIC_DIR

    idx = os.path.join(STATIC_DIR, "index.html")
    assert os.path.exists(idx), f"缺少 {idx}"
    content = open(idx, encoding="utf-8").read()
    assert "RouteLLM" in content
    # 应引用图表库（ECharts）
    assert "echarts" in content.lower()
