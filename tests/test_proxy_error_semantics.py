"""转发层的错误传播语义（TDD RED 阶段）。

问题
----
面板转发层把网关返回的**所有** `>=400` 都转成 `HTTP 200 + proxy_error`
包装。设计初衷是"面板不因网关异常整页崩"，对 5xx / 不可达是对的；
但用于 **4xx 客户端错误**就错了：

    网关返回 422（参数非法）
      → 面板对前端报 HTTP 200
      → 前端无法区分"成功"与"参数错"

实测暴露路径：
    POST /api/config/verify?tier=medium   （非法 tier）
    网关 → 422 {"detail": "未知 tier: 'medium'"}
    面板 → HTTP 200 {"proxy_error": "网关返回 HTTP 422", ...}

修正原则
--------
    4xx 客户端错误 → **保留状态码**（透传，前端才能正确分支）
    5xx 服务端错误 → 归一化成 200 + proxy_error（既有降级设计，保持不变）
    网关不可达     → 归一化成 200 + proxy_error（同上）

运行：
    pytest tests/test_proxy_error_semantics.py -v
"""

import asyncio

import pytest


@pytest.fixture
def proxy():
    import routellm.monitoring.dashboard.config_proxy as m

    return m


def _patch(proxy, monkeypatch, result):
    """把 _request 换成返回固定结果的假函数。"""

    async def fake_request(method, path, body=None, timeout=15.0):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(proxy, "_request", fake_request)


# ================================================ 4xx：保留状态码


@pytest.mark.parametrize("method_name", ["forward_get_config", "forward_verify"])
def test_get_and_verify_propagate_4xx(proxy, monkeypatch, method_name):
    """GET / verify 收到 4xx 须抛 HTTPException（保留状态码）。"""
    from fastapi import HTTPException

    _patch(proxy, monkeypatch, (422, {"detail": "未知 tier"}))
    fn = getattr(proxy, method_name)
    with pytest.raises(HTTPException) as e:
        asyncio.run(fn())
    assert e.value.status_code == 422


def test_put_propagates_4xx(proxy, monkeypatch):
    """PUT 收到 4xx 须抛 HTTPException。"""
    from fastapi import HTTPException

    _patch(proxy, monkeypatch, (422, {"detail": "unknown field"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(proxy.forward_put_config({"bad": 1}))
    assert e.value.status_code == 422


def test_401_propagates_so_frontend_can_react(proxy, monkeypatch):
    """401 须透传（前端据此提示"检查网关 key"，而非当作成功）。"""
    from fastapi import HTTPException

    _patch(proxy, monkeypatch, (401, {"detail": "unauthorized"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(proxy.forward_get_config())
    assert e.value.status_code == 401


def test_4xx_detail_is_preserved(proxy, monkeypatch):
    """4xx 的原始错误详情须带给前端（不要吞成笼统文案）。"""
    from fastapi import HTTPException

    _patch(proxy, monkeypatch, (422, {"detail": "未知 tier: 'medium'"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(proxy.forward_verify(tier="medium"))
    assert "medium" in str(e.value.detail)


# ================================================ 5xx / 不可达：保持归一化


def test_5xx_still_normalized(proxy, monkeypatch):
    """5xx 仍归一化成 200 + proxy_error（面板不崩，既有降级设计）。"""
    _patch(proxy, monkeypatch, (503, {"detail": "boom"}))
    res = asyncio.run(proxy.forward_verify(tier="strong"))
    assert res.get("proxy_error")
    assert res.get("proxy_status") == 503


def test_unreachable_still_normalized(proxy, monkeypatch):
    """网关不可达仍归一化。"""
    _patch(proxy, monkeypatch, ConnectionError("refused"))
    res = asyncio.run(proxy.forward_get_config())
    assert res.get("proxy_error")


def test_5xx_on_put_still_normalized(proxy, monkeypatch):
    """PUT 的 5xx 也不抛（保持既有行为）。"""
    _patch(proxy, monkeypatch, (500, {"detail": "boom"}))
    res = asyncio.run(proxy.forward_put_config({"strong_model": "x"}))
    assert res.get("proxy_error")


# ================================================ 成功路径不受影响


def test_success_path_unchanged(proxy, monkeypatch):
    """成功仍返回数据本身（保留 proxy_error=None 的既有约定）。"""
    _patch(proxy, monkeypatch, (200, {"strong_model": "m"}))
    res = asyncio.run(proxy.forward_get_config())
    assert res["strong_model"] == "m"
    assert res.get("proxy_error") is None


def test_verify_success_path_unchanged(proxy, monkeypatch):
    """verify 成功时原样返回网关数据（含 probed 结构）。"""
    payload = {"ok": True, "tier": "weak", "probed": {"model": "w"}}
    _patch(proxy, monkeypatch, (200, payload))
    res = asyncio.run(proxy.forward_verify(tier="weak"))
    assert res == payload
