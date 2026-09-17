"""监控体系。

设计依据：方案文档 4.2 节（P0）—— 方案 B + A 混合
    自研 Dashboard（FastAPI + SQLite + 单页 HTML/ECharts）
    + Prometheus 文本端点（兼容标准生态）

模块：
    metrics.py     指标数据模型 + 成本估算
    store.py       SQLite 持久化（解决原实现「重启丢失」问题）
    prometheus.py  Prometheus 文本格式（零依赖自实现）
    middleware.py  FastAPI 中间件（请求级采集）
    dashboard/     自研可视化面板（页面 + API）
"""

from routellm.monitoring.metrics import RequestMetrics, estimate_cost
from routellm.monitoring.middleware import MetricsMiddleware
from routellm.monitoring.store import MetricsStore

__all__ = [
    "RequestMetrics",
    "estimate_cost",
    "MetricsMiddleware",
    "MetricsStore",
]
