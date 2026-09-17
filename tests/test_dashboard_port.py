"""Dashboard 独立端口暴露的单元测试（TDD RED 阶段）。

需求：Dashboard 与网关**职责分离、端口差异化**：
    6060  网关业务路由（/v1/chat/completions 等），仅绑回环
    8091  监控面板（/dashboard + /api/metrics/*），绑 0.0.0.0 供外部访问

设计选择：**同进程、双端口**而非独立容器
    理由：Dashboard 需要的指标数据在网关进程的 SQLite 里（共享文件亦可），
    且 metrics 中间件产生的 Prometheus 计数器是**进程内内存态** ——
    独立容器读不到（除非全走 PromQL）。同进程双端口保持了简单性，
    同时达成「端口差异化、职责分离」的目标。

    仅暴露面板相关路由到 8091：业务路由（/v1/*）不对外开第二个入口。

运行：
    pytest tests/test_dashboard_port.py -v
"""

import os

import pytest



def _all_paths(app):
    """收集 app 的所有路由路径。

    新版 FastAPI 的 app.routes 里含 _IncludedRouter 包装对象
    —— 它没有 .path，路由挂在 .original_router.routes 上，需展开。
    （实测：_IncludedRouter 既无 .path 也无 .routes，只有
      .original_router —— 故不能只判断 hasattr(r, "routes")）
    """
    out = set()
    for r in app.routes:
        path = getattr(r, "path", None)
        if path is not None:
            out.add(path)
        else:
            orig = getattr(r, "original_router", None)
            if orig is not None:
                for inner in getattr(orig, "routes", []):
                    p = getattr(inner, "path", None)
                    if p is not None:
                        out.add(p)
    return out


def test_dashboard_app_has_dashboard_routes():
    """面板 app 须含面板与图表 API 路由。"""
    from routellm.monitoring.dashboard.app import create_dashboard_only_app

    app = create_dashboard_only_app()
    paths = _all_paths(app)
    for p in ("/", "/dashboard", "/api/metrics/summary", "/api/metrics/recent",
              "/api/metrics/timeseries", "/health"):
        assert p in paths, f"缺少路由 {p}"


def test_dashboard_app_excludes_business_routes():
    """面板 app 不得暴露业务路由（职责分离）。"""
    from routellm.monitoring.dashboard.app import create_dashboard_only_app

    app = create_dashboard_only_app()
    paths = _all_paths(app)
    assert "/v1/chat/completions" not in paths, "面板端口不应暴露业务接口"
    assert "/v1/models" not in paths


def test_dashboard_port_default_is_8080():
    """默认面板端口须为 8080（容器内端口；宿主机端口由 compose 映射决定）。

    与网关 6060 差异化即可 —— 容器内端口互不冲突（不同网络命名空间）。
    """
    from routellm.monitoring.dashboard.app import DASHBOARD_PORT

    assert DASHBOARD_PORT == 8080


def test_dashboard_bind_host_default():
    """默认绑定 0.0.0.0（外部可访问）。"""
    from routellm.monitoring.dashboard.app import DASHBOARD_HOST

    assert DASHBOARD_HOST == "0.0.0.0"


def test_port_configurable_by_env(monkeypatch):
    """端口与绑定地址须可由环境变量覆盖（compose/部署时用）。"""
    m = __import__("routellm.monitoring.dashboard.app", fromlist=["x"])

    monkeypatch.setenv("ROUTELLM_DASHBOARD_PORT", "9999")
    monkeypatch.setenv("ROUTELLM_DASHBOARD_HOST", "127.0.0.1")
    assert m.get_dashboard_port() == 9999
    assert m.get_dashboard_host() == "127.0.0.1"


def test_panel_start_failure_returns_none_not_raise():
    """面板端口被占时必须返回 None 而非抛异常。

    这是个重要的不变量：面板是辅助功能，其启动失败绝不能中断网关的
    lifespan（实测踩过：面板 bind 失败抛错 → lifespan 中断 →
    CONTROLLER 未初始化 → 业务全部 500）。
    """
    import socket

    from routellm.monitoring.dashboard.app import start_dashboard_server_in_thread

    # 占住一个端口，再让面板去绑同一个
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    busy_port = sock.getsockname()[1]
    try:
        t = start_dashboard_server_in_thread(host="127.0.0.1", port=busy_port)
        assert t is None, "端口被占时应返回 None（不抛异常、不启线程）"
    finally:
        sock.close()


def test_panel_start_success_returns_thread():
    """端口可用时应正常返回已启动的线程。"""
    import socket

    from routellm.monitoring.dashboard.app import start_dashboard_server_in_thread

    # 找一个空闲端口
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()

    t = start_dashboard_server_in_thread(host="127.0.0.1", port=free_port)
    assert t is not None and t.daemon is True


def test_dashboard_app_shares_store(store, monkeypatch):
    """面板 app 须复用同一个 store（与网关同进程共享数据源）。"""
    m = __import__("routellm.monitoring.dashboard.app", fromlist=["x"])

    m.set_store(store)
    app = m.create_dashboard_only_app()
    assert app is not None
    assert m.get_store() is store


def test_main_entry_parses_args(monkeypatch):
    """main 须支持 --port/--host/--db（供 uvicorn 命令行启动）。"""
    import inspect

    m = __import__("routellm.monitoring.dashboard.app", fromlist=["x"])

    src = inspect.getsource(m.main)
    for flag in ("--port", "--host", "--db"):
        assert flag in src, f"main 缺少参数 {flag}"


@pytest.fixture
def store(tmp_path):
    from routellm.monitoring.store import MetricsStore

    db = str(tmp_path / "m.db")
    s = MetricsStore(db)
    yield s
    # 清理全局单例，避免污染其他测试
    m = __import__("routellm.monitoring.dashboard.app", fromlist=["x"])

    m._STORE = None
