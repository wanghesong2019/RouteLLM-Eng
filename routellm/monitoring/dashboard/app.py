"""Dashboard 独立端口服务（同进程双端口，职责分离）。

需求
----
Dashboard 与网关**端口差异化、各是各的职责**：

    6060  网关业务（/v1/chat/completions、/v1/models 等）
          默认绑 127.0.0.1（仅本机 / 经反代访问）
    8091  监控面板（/dashboard、/api/metrics/*）
          默认绑 0.0.0.0（供外部浏览器访问）

设计选择：**同进程、双端口**（而非独立容器）
------------------------------------------
理由：
    1. 指标数据源是网关进程写的 SQLite，同进程可直接共享 store 实例
    2. Prometheus 计数器（routellm.monitoring.prometheus）是**进程内内存态**，
       独立容器读不到 —— 除非把聚合全部改走 PromQL（另一个改造项）
    3. 达成「端口差异化、职责分离」的目标，同时不引入跨进程数据一致性问题

    若将来需要独立扩容，可把本模块单独起容器（`python -m
    routellm.monitoring.dashboard.app`），但那时需改为从 Prometheus 读数据。

安全提示
--------
默认绑 0.0.0.0 意味着**面板对外开放**。本服务**没有鉴权** ——
部署时应依赖外层防火墙/安全组限制来源，或在前置反代上加认证。
可用 ROUTELLM_DASHBOARD_HOST 覆盖为 127.0.0.1 收回本机。

启动：
    # 独立进程（开发/调试）
    python -m routellm.monitoring.dashboard.app --port 8091 --db /data/metrics.db

    # 同进程双端口（生产，由 openai_server.py 启动）
    # 见 routellm/openai_server.py 的 _start_dashboard_server()
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from routellm.monitoring.store import MetricsStore

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 默认端口/绑定。
# 端口 8080 是**容器内**端口，由 compose 映射到宿主机（如 8092:8080）——
# 容器内端口与网关 6060 不同即可，宿主机端口按实际情况定。
DASHBOARD_PORT = 8080
DASHBOARD_HOST = "0.0.0.0"

_STORE: Optional[MetricsStore] = None

router = APIRouter()


# --------------------------------------------------------------------------- 配置


def get_dashboard_port() -> int:
    """面板端口（环境变量 ROUTELLM_DASHBOARD_PORT 可覆盖）。"""
    return int(os.environ.get("ROUTELLM_DASHBOARD_PORT", DASHBOARD_PORT))


def get_dashboard_host() -> str:
    """面板绑定地址（ROUTELLM_DASHBOARD_HOST 可覆盖为 127.0.0.1）。"""
    return os.environ.get("ROUTELLM_DASHBOARD_HOST", DASHBOARD_HOST)


# --------------------------------------------------------------------------- store


def set_store(store: MetricsStore) -> None:
    """注入 store（网关启动时传入共享实例；测试也用它）。"""
    global _STORE
    _STORE = store


def get_store() -> MetricsStore:
    """取 store。惰性创建，路径由 ROUTELLM_METRICS_DB 决定。"""
    global _STORE
    if _STORE is None:
        db = os.environ.get("ROUTELLM_METRICS_DB", "/data/metrics.db")
        _STORE = MetricsStore(db)
        logger.info("Dashboard 数据源: %s", db)
    return _STORE


# --------------------------------------------------------------------------- 页面


def _read_index() -> Optional[str]:
    idx = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(idx):
        return None
    with open(idx, encoding="utf-8") as f:
        return f.read()


@router.get("/", response_class=HTMLResponse)
async def root_page() -> HTMLResponse:
    """根路径即面板（方便直接访问 http://host:8091/）。"""
    return await dashboard_page()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> HTMLResponse:
    """单页 Dashboard。"""
    html = _read_index()
    if html is None:
        return HTMLResponse(
            "<h1>Dashboard 静态文件缺失</h1><p>期望: " + STATIC_DIR + "/index.html</p>",
            status_code=500,
        )
    return HTMLResponse(html)


@router.get("/config", response_class=HTMLResponse)
async def config_page() -> HTMLResponse:
    """配置编辑页（运行时配置热更新，方案文档 4.8）。

    这是独立页面（而非塞进 index.html）—— 配置编辑与监控展示是两种
    不同的操作场景，分开更清晰，也便于单独鉴权/审计。
    """
    p = os.path.join(STATIC_DIR, "config.html")
    if not os.path.exists(p):
        return HTMLResponse(
            "<h1>配置页静态文件缺失</h1><p>期望: " + p + "</p>", status_code=500
        )
    with open(p, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# --------------------------------------------------------------------------- API


@router.get("/api/metrics/summary")
async def api_summary() -> Dict[str, Any]:
    """聚合指标（卡片 + 图表）。"""
    return await get_store().summary()


@router.get("/api/metrics/recent")
async def api_recent(limit: int = 20) -> Dict[str, Any]:
    """最近请求列表。"""
    try:
        rows = await get_store().recent(limit=limit)
        return {"count": len(rows), "requests": rows}
    except Exception as e:  # noqa: BLE001
        logger.warning("读取最近请求失败: %s", e)
        return {"count": 0, "requests": [], "error": f"{type(e).__name__}: {e}"}


@router.get("/api/metrics/timeseries")
async def api_timeseries(bucket_seconds: float = 60.0) -> Dict[str, Any]:
    """时间序列（折线图）。"""
    try:
        series = await get_store().time_series(bucket_seconds=bucket_seconds)
        return {"bucket_seconds": bucket_seconds, "series": series}
    except Exception as e:  # noqa: BLE001
        logger.warning("读取时序失败: %s", e)
        return {"bucket_seconds": bucket_seconds, "series": [], "error": f"{type(e).__name__}: {e}"}


@router.get("/health")
async def health() -> Dict[str, Any]:
    """健康检查（含数据源可达性，供探针用）。"""
    store = get_store()
    info: Dict[str, Any] = {
        "status": "online",
        "service": "routellm-dashboard",
        "db_path": store.db_path,
        "port": get_dashboard_port(),
    }
    try:
        s = await store.summary()
        info["total_requests"] = s["total_requests"]
        info["db_ok"] = True
    except Exception as e:  # noqa: BLE001
        info["db_ok"] = False
        info["db_error"] = f"{type(e).__name__}: {e}"
    return info


# --------------------------------------------------------------------------- app


def create_dashboard_only_app() -> FastAPI:
    """构造**只含面板路由**的 app（不含业务接口）。

    这是绑定 8091 的那个 app —— 刻意不挂业务路由，
    实现「职责分离」：面板端口只做面板的事。
    """
    app = FastAPI(
        title="RouteLLM Dashboard",
        description="自研路由监控面板（独立端口 8091）",
        version="0.1.0",
    )
    app.include_router(router)

    # 配置转发路由（方案文档 4.8）—— 把编辑请求转发到网关
    try:
        from routellm.monitoring.dashboard.config_proxy import router as _cfg_router

        app.include_router(_cfg_router)
    except Exception as e:  # noqa: BLE001
        logger.warning("配置转发路由挂载失败: %s", e)

    return app


# 模块级 app 实例（uvicorn 加载用）
app = create_dashboard_only_app()


def start_dashboard_server_in_thread(
    store: Optional[MetricsStore] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Optional[threading.Thread]:
    """在后台线程启动面板服务（供网关同进程双端口使用）。

    返回已启动的线程；**端口不可用等启动失败时返回 None 而不抛异常** ——
    面板是辅助功能，其失败绝不能影响网关主服务（实测踩过：面板 bind
    失败抛错中断了 lifespan，导致 CONTROLLER 未初始化、业务全部 500）。

    实现要点：先用 socket 预检端口可用性（在主线程内完成，失败可即时
    返回 None），确认可用后再交给子线程跑 uvicorn —— 避免"异步失败"
    无法捕获的问题。
    """
    import socket

    host = host or get_dashboard_host()
    port = port or get_dashboard_port()

    # 端口预检（同容器内可能被 docker-proxy 等占用）
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as e:
        logger.warning(
            "面板端口 %s:%s 不可用（%s）—— 跳过面板服务，网关不受影响",
            host, port, e,
        )
        return None
    finally:
        probe.close()

    if store is not None:
        set_store(store)

    server_app = create_dashboard_only_app()

    def _run() -> None:
        import uvicorn

        try:
            logger.info("Dashboard 服务启动于 %s:%s", host, port)
            uvicorn.run(server_app, host=host, port=port, log_level="warning")
        except Exception as e:  # noqa: BLE001
            # 兜底：线程内任何异常都只记日志，不影响主进程
            logger.warning("Dashboard 服务异常退出（网关不受影响）: %s", e)

    t = threading.Thread(target=_run, name="routellm-dashboard", daemon=True)
    t.start()
    return t


def main() -> None:
    """独立进程启动入口（开发/调试用）。"""
    import uvicorn

    ap = argparse.ArgumentParser(description="RouteLLM Dashboard 面板服务")
    ap.add_argument("--host", default=None, help=f"绑定地址（默认 {DASHBOARD_HOST}）")
    ap.add_argument("--port", type=int, default=None, help=f"端口（默认 {DASHBOARD_PORT}）")
    ap.add_argument("--db", default=None, help="metrics SQLite 路径（覆盖环境变量）")
    args = ap.parse_args()

    if args.db:
        os.environ["ROUTELLM_METRICS_DB"] = args.db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = args.host or get_dashboard_host()
    port = args.port or get_dashboard_port()
    logger.info("Dashboard 启动: %s:%s (db=%s)", host, port, get_store().db_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
