"""监控体系。

设计依据：方案文档 4.2 节（P0）—— 方案 B + A 混合
    自研 Dashboard（FastAPI + SQLite + 单页 HTML/ECharts）
    + Prometheus 文本端点（兼容标准生态）

模块：
    metrics.py     指标数据模型 + 成本估算
    store.py       SQLite 持久化（解决原实现「重启丢失」问题）
    prometheus.py  Prometheus 文本格式（零依赖自实现）
    middleware.py  FastAPI 中间件（请求级采集）
    auth.py        网关 API key 鉴权
    dashboard/     自研可视化面板（页面 + API）

导入约定（重要）
----------------
本 `__init__.py` **刻意不做子模块的热切导入**（no eager re-export），
原因：Dashboard 是独立容器，只装了 FastAPI + uvicorn（不含 litellm 等
网关重依赖）。若在此处 `from .middleware import ...`，则 Dashboard 容器
import 本包时会连带拉起 middleware → litellm，直接 ImportError
（实测踩过：`ModuleNotFoundError: No module named 'routellm.monitoring.middleware'`）。

因此各子模块须**显式按路径导入**，例如：
    from routellm.monitoring.store import MetricsStore
    from routellm.monitoring.metrics import RequestMetrics
    from routellm.monitoring.auth import ApiKeyMiddleware      # 仅网关需要
    from routellm.monitoring.middleware import MetricsMiddleware  # 仅网关需要
"""

__all__ = [
    "metrics",
    "store",
    "prometheus",
    "middleware",
    "auth",
    "dashboard",
]
