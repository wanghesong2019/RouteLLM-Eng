"""Dashboard：自研可视化面板。

设计依据：方案文档 4.2 节（方案 B + A 混合）
    - 自研 Dashboard 展示全栈能力（后端 API + 前端可视化）
    - 同时暴露 Prometheus 端点兼容标准生态

为什么自研而非 Grafana：
    RouteLLM 的核心价值是「省钱」—— 自研面板可直接展示成本节省曲线，
    比配置 Grafana 面板更直观，且无需额外部署组件。

路由：
    GET /dashboard                   单页 HTML
    GET /api/metrics/summary         聚合指标（卡片 + 图表数据）
    GET /api/metrics/recent          最近请求列表
    GET /api/metrics/timeseries      时间序列（折线图）
    GET /metrics                     Prometheus 文本格式
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

from routellm.monitoring import prometheus
from routellm.monitoring.store import MetricsStore

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_STORE: Optional[MetricsStore] = None

router = APIRouter()


def set_store(store: MetricsStore) -> None:
    """注入 store（由网关在启动时调用；测试也用它）。"""
    global _STORE
    _STORE = store


def get_store() -> MetricsStore:
    """取 store；未注入时惰性建一个内存库（避免 /dashboard 直接 500）。"""
    global _STORE
    if _STORE is None:
        db = os.environ.get("ROUTELLM_METRICS_DB", ":memory:")
        _STORE = MetricsStore(db)
    return _STORE


# ------------------------------------------------------------------ 页面


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> HTMLResponse:
    """单页 Dashboard。"""
    idx = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(idx):
        return HTMLResponse(
            "<h1>Dashboard 静态文件缺失</h1>"
            f"<p>期望路径: {idx}</p>",
            status_code=500,
        )
    with open(idx, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ------------------------------------------------------------------ API


@router.get("/api/metrics/summary")
async def api_summary() -> Dict[str, Any]:
    """聚合指标（顶部卡片 + 饼图等）。"""
    return await get_store().summary()


@router.get("/api/metrics/recent")
async def api_recent(limit: int = 20) -> Dict[str, Any]:
    """最近请求列表。"""
    rows = await get_store().recent(limit=limit)
    return {"count": len(rows), "requests": rows}


@router.get("/api/metrics/timeseries")
async def api_timeseries(bucket_seconds: float = 60.0) -> Dict[str, Any]:
    """时间序列（折线图数据）。"""
    series = await get_store().time_series(bucket_seconds=bucket_seconds)
    return {"bucket_seconds": bucket_seconds, "series": series}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus 文本格式指标。"""
    return PlainTextResponse(prometheus.render_metrics())
