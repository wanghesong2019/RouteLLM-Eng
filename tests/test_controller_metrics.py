"""Controller 暴露路由元信息的单元测试（TDD RED 阶段）。

问题（端到端验证发现）：
    监控体系需要「路由置信度 win_rate」与「路由计算延迟」两个指标，
    但 Controller._get_routed_model_for_completion 只返回模型名，
    把 win_rate 与耗时都丢掉了 —— 导致 Dashboard 里 win_rate 显示 N/A、
    routing_latency_ms 恒为 0。

设计：
    给 Controller 增加一个「最近一次路由元信息」的读取接口
    （`last_routing_info`），在完成路由决策时记录：
        {router, threshold, win_rate, routed_model, latency_ms}
    网关在请求处理完从 Controller 读取（同一协程内，无并发串台问题
    —— 异步 Controller 每个请求实例独立？需确认；故用 contextvar 更稳）。

    为并发安全，元信息存于 contextvars.ContextVar（按协程隔离），
    而非实例属性 —— 否则并发请求会互相覆盖。

运行：
    pytest tests/test_controller_metrics.py -v
"""

import pytest


def test_controller_exposes_routing_info():
    """Controller 须提供读取最近路由元信息的接口。"""
    from routellm.controller import Controller

    assert hasattr(Controller, "last_routing_info"), (
        "Controller 缺少 last_routing_info —— 监控需要 win_rate/延迟"
    )


def test_routing_info_shape():
    """路由元信息须含监控所需字段。"""
    from routellm.controller import RoutingInfo

    fields = set(RoutingInfo.__dataclass_fields__)
    for f in ("router", "threshold", "win_rate", "routed_model", "latency_ms"):
        assert f in fields, f"RoutingInfo 缺少字段 {f}"


def test_routing_info_recorded(monkeypatch):
    """完成一次路由决策后，last_routing_info 应可读出。"""
    from routellm import controller as ctrl_mod

    # 用最小 Controller 实例（不走真实 LLM）
    c = ctrl_mod.Controller.__new__(ctrl_mod.Controller)

    info = ctrl_mod.RoutingInfo(
        router="remote_bert",
        threshold=0.5,
        win_rate=0.42,
        routed_model="strong",
        latency_ms=6.1,
    )
    ctrl_mod._set_routing_info(info)

    got = ctrl_mod.get_routing_info()
    assert got is not None
    assert got.win_rate == pytest.approx(0.42)
    assert got.routed_model == "strong"
    assert got.latency_ms == pytest.approx(6.1)


def test_routing_info_context_isolated():
    """路由元信息须按上下文隔离（并发请求不串台）。

    用 contextvars.copy_context 模拟两个并发上下文，
    验证各自 set 互不影响。
    """
    import contextvars

    from routellm import controller as ctrl_mod

    def make(wr):
        return ctrl_mod.RoutingInfo(
            router="r", threshold=0.5, win_rate=wr,
            routed_model="weak", latency_ms=1.0,
        )

    ctx1 = contextvars.copy_context()
    ctx2 = contextvars.copy_context()

    ctx1.run(ctrl_mod._set_routing_info, make(0.11))
    ctx2.run(ctrl_mod._set_routing_info, make(0.99))

    assert ctx1.run(ctrl_mod.get_routing_info).win_rate == pytest.approx(0.11)
    assert ctx2.run(ctrl_mod.get_routing_info).win_rate == pytest.approx(0.99)


def test_routing_info_cleared_between_requests():
    """每次设置应覆盖上一次（不残留）。"""
    from routellm import controller as ctrl_mod

    ctrl_mod._set_routing_info(
        ctrl_mod.RoutingInfo("r", 0.5, 0.1, "weak", 1.0)
    )
    ctrl_mod._set_routing_info(
        ctrl_mod.RoutingInfo("r", 0.5, 0.9, "strong", 2.0)
    )
    got = ctrl_mod.get_routing_info()
    assert got.win_rate == pytest.approx(0.9)
    assert got.routed_model == "strong"
