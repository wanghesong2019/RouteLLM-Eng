"""monitoring 包初始化不得拉起重依赖（TDD 守护测试）。

背景
----
Dashboard 是独立容器，只装 FastAPI + uvicorn（**不含** litellm / pandas /
sklearn 等网关依赖）。若 `routellm/monitoring/__init__.py` 做热切导入
（如 `from .middleware import MetricsMiddleware`），则 Dashboard 容器
import 本包时会连带拉起 middleware → litellm → ImportError。

实测踩过：Dashboard 容器启动即
    ModuleNotFoundError: No module named 'routellm.monitoring.middleware'

本测试守护该不变量：仅导入 `routellm.monitoring` 包时，不应触发
middleware / auth / dashboard 等子模块的导入。

运行：
    pytest tests/test_monitoring_package_lazy.py -v
"""

import subprocess
import sys


def test_importing_package_does_not_load_heavy_submodules():
    """只 import 包时，middleware/auth/dashboard 不应被加载。

    用子进程验证（避免当前测试进程已导入其他模块造成干扰）。
    """
    code = (
        "import sys;"
        # 隔离：先屏蔽可能存在的 litellm，模拟 Dashboard 容器的环境
        "sys.modules['litellm'] = None;"
        "import routellm.monitoring;"
        "loaded = [m for m in ('routellm.monitoring.middleware',"
        " 'routellm.monitoring.auth',"
        " 'routellm.monitoring.dashboard') if m in sys.modules];"
        "print('LOADED:' + ','.join(loaded))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=".",
    )
    assert r.returncode == 0, f"导入包失败: {r.stderr[-500:]}"
    loaded = r.stdout.strip().replace("LOADED:", "")
    assert loaded == "", (
        f"包初始化拉起了子模块: {loaded} —— "
        "这会让轻量消费者（Dashboard 容器）被迫加载网关重依赖"
    )


def test_store_and_metrics_importable_without_gateway_deps():
    """store / metrics 须能在无网关依赖的环境下导入。

    这两个是 Dashboard 容器唯一需要的 monitoring 子模块。
    """
    code = (
        "import sys;"
        "sys.modules['litellm'] = None;"
        "sys.modules['pandas'] = None;"
        "from routellm.monitoring.store import MetricsStore;"
        "from routellm.monitoring.metrics import RequestMetrics, estimate_cost;"
        "print('OK')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=".",
    )
    assert r.returncode == 0, f"轻量导入失败: {r.stderr[-500:]}"
    assert "OK" in r.stdout
