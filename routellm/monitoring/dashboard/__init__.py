"""Dashboard 包 —— 路由监控面板。

职责分离（端口差异化）
   6060  网关业务路由（/v1/chat/completions 等）
   8091  监控面板（/dashboard、/api/metrics/*），绑 0.0.0.0 供外部访问

使用方式
--------
同进程双端口（生产，由 openai_server.py 启动）：
    from routellm.monitoring.dashboard import (
        router, set_store, start_dashboard_server_in_thread,
    )
    set_store(metrics_store)
    app.include_router(router)                     # 可选：网关端口也提供面板
    start_dashboard_server_in_thread(metrics_store)  # 8091 独立端口

独立进程（开发/调试）：
    python -m routellm.monitoring.dashboard.app --port 8091 --db /data/metrics.db

安全：面板无鉴权。绑 0.0.0.0 时须依赖外层防火墙/安全组限制来源，
或用 ROUTELLM_DASHBOARD_HOST=127.0.0.1 收回本机。
"""

from routellm.monitoring.dashboard.app import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    STATIC_DIR,
    create_dashboard_only_app,
    get_dashboard_host,
    get_dashboard_port,
    get_store,
    router,
    set_store,
    start_dashboard_server_in_thread,
)
# 注意：不要在此处 `from ...app import app` —— 子模块名为 app，
# 同名导入会遮蔽子模块本身（`routellm.monitoring.dashboard.app`
# 会变成 FastAPI 实例而非模块），导致后续 __import__ 解析错误。
# 需要 app 实例时用 create_dashboard_only_app()，或用别名：
from routellm.monitoring.dashboard.app import app as dashboard_app

__all__ = [
    "dashboard_app",
    "router",
    "set_store",
    "get_store",
    "create_dashboard_only_app",
    "start_dashboard_server_in_thread",
    "get_dashboard_host",
    "get_dashboard_port",
    "STATIC_DIR",
    "DASHBOARD_HOST",
    "DASHBOARD_PORT",
]
