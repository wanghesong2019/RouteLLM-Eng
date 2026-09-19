"""面板端点的控制器注入测试（回归：双模块实例陷阱）。

背景（线上实测踩到）
--------------------
`python -m routellm.openai_server` 启动时，跑 lifespan 的是 `__main__` 那份
模块；而面板端点若写 `from routellm.openai_server import _ADAPTIVE_THRESHOLD`，
拿到的是 **另一份**从未跑过 lifespan 的副本 —— 该副本里恒为 None，于是
`/api/adaptive-threshold` 永远返回 `enabled:false`，看起来像"功能没开启"。

修法：由网关在 lifespan 里把控制器**注入**面板（set_adaptive_threshold），
面板不反向 import 网关模块。

契约：本模块导入面板应用**不得**把 openai_server 拉进 sys.modules ——
这是"没有反向依赖"的可检验形式。
"""

import sys

import pytest


def test_dashboard_import_does_not_pull_openai_server():
    """导入面板路由不应连带导入 openai_server（否则双实例陷阱会复现）。"""
    # 先清掉可能已加载的 openai_server（测试进程里可能被别的用例导入过）
    for name in list(sys.modules):
        if name == "routellm.openai_server":
            del sys.modules[name]

    from routellm.monitoring.dashboard import app as dash_app  # noqa: F401

    assert "routellm.openai_server" not in sys.modules, (
        "面板模块反向依赖了 openai_server —— python -m 启动时会拿到未初始化"
        "的另一份副本，端点将恒报 enabled=false"
    )


def test_adaptive_endpoint_uses_injected_controller():
    """注入后端点应报告真实状态（而非 enabled=false）。"""
    from routellm.routers.adaptive_threshold import (
        AdaptiveThresholdConfig,
        AdaptiveThresholdController,
    )
    from routellm.monitoring.dashboard import set_adaptive_threshold, router

    ctrl = AdaptiveThresholdController(
        AdaptiveThresholdConfig(tau_base=0.5, tau_max=0.75, budget_tokens_per_min=1234),
        metrics_store=None,
    )
    set_adaptive_threshold(ctrl)

    # 找到端点函数直接调用（避免起 HTTP 栈）
    fn = None
    for r in router.routes:
        if getattr(r, "path", "") == "/api/adaptive-threshold":
            fn = r.endpoint
            break
    assert fn is not None, "端点未注册"

    import asyncio

    st = asyncio.run(fn())
    assert st["enabled"] is True
    assert st["tau_max"] == 0.75
    assert st["budget_tokens_per_min"] == 1234
    assert st["source"] == "in-process"

    # 清理，避免影响其他用例
    set_adaptive_threshold(None)


def test_adaptive_endpoint_reports_unavailable_without_injection(monkeypatch):
    """未注入且无网关地址时应显式报 unavailable（而不是假装 enabled=false）。"""
    from routellm.monitoring.dashboard import set_adaptive_threshold, router

    set_adaptive_threshold(None)
    monkeypatch.delenv("ROUTELLM_GATEWAY_URL", raising=False)

    fn = None
    for r in router.routes:
        if getattr(r, "path", "") == "/api/adaptive-threshold":
            fn = r.endpoint
            break

    import asyncio

    st = asyncio.run(fn())
    assert st["enabled"] is False
    assert st["source"] == "unavailable"
