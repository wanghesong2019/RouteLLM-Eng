"""server 层容错行为测试（方案文档 4.3 步骤 7）。

Controller 的降级链产生两类「非正常但需正确表达」的结果，server 层必须
把它们翻译成恰当的 HTTP 语义：

1. FallbackExhaustedError → 503 + `Retry-After: 30`
   503 而非 500：这是**暂时性**上游不可用，客户端应当重试。
   500 会让客户端误以为是自身请求有问题。

2. 降级发生 → 响应头 `X-RouteLLM-Downgraded: true`
   否则客户端会把「弱模型的回答」当成原始路由结果，无法察觉质量变化。

同时必须保证：正常路径与既有 4xx 语义不受影响（回归）。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import routellm.openai_server as srv
from routellm.resilience import FallbackExhaustedError


class _FakeResp:
    """litellm 响应替身，字段满足 ChatCompletionResponse 的构造需求。"""

    def __init__(self, text: str = "ok") -> None:
        self.id = "chatcmpl-test"
        self.object = "chat.completion"
        self.created = 1
        self.model = "m"
        self.usage = type(
            "U", (), {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2}
        )()
        self.choices = [type("C", (), {
            "index": 0,
            "message": type("M", (), {"role": "assistant", "content": text})(),
            "finish_reason": "stop",
        })()]

    def model_dump(self):
        return {
            "id": self.id, "object": self.object, "created": self.created,
            "model": self.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }


@pytest.fixture
def client(monkeypatch):
    """构造把 CONTROLLER 替换成可控替身的 TestClient。

    不使用 `with TestClient(...)` 上下文管理器：那会触发 lifespan，而
    lifespan 会构造**真实** Controller（含 config_store 与路由器加载），
    对响应层测试是无关负担，还会因配置/权重缺失而失败。
    本文件只验证「异常 → HTTP 语义」的翻译，直接构造 app 即可。

    但路由器的依赖（`app.post` 装饰器、中间件）已在导入期完成，
    替换模块级 CONTROLLER 即可生效。
    """
    class _FakeController:
        last_downgraded = False

        def __init__(self):
            self.behavior = "ok"

        async def acompletion(self, **kwargs):
            if self.behavior == "exhausted":
                self.last_downgraded = True
                raise FallbackExhaustedError(retry_after=30)
            if self.behavior == "downgraded":
                self.last_downgraded = True
                return _FakeResp("weak")
            self.last_downgraded = False
            return _FakeResp("strong")

        def completion(self, **kwargs):
            return _FakeResp("sync")

    fc = _FakeController()
    monkeypatch.setattr(srv, "CONTROLLER", fc)
    yield TestClient(srv.app), fc


def _payload(model="router-random-0.5"):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_exhausted_upstream_returns_503_with_retry_after(client):
    """降级链全耗尽 → 503 + Retry-After（而非 500）。"""
    c, fc = client
    fc.behavior = "exhausted"

    r = c.post("/v1/chat/completions", json=_payload())
    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"
    assert r.headers.get("retry-after") == "30"


def test_downgraded_response_sets_header(client):
    """降级响应必须带 X-RouteLLM-Downgraded: true。"""
    c, fc = client
    fc.behavior = "downgraded"

    r = c.post("/v1/chat/completions", json=_payload())
    assert r.status_code == 200
    assert r.headers.get("x-routellm-downgraded") == "true"


def test_normal_response_has_no_downgraded_header(client):
    """正常路径不应出现降级头（避免客户端误判）。"""
    c, fc = client
    fc.behavior = "ok"

    r = c.post("/v1/chat/completions", json=_payload())
    assert r.status_code == 200
    assert r.headers.get("x-routellm-downgraded") in (None, "false")


def test_routing_error_still_400(client):
    """既有语义回归：RoutingError 仍是 400（未被容错改动影响）。"""
    c, fc = client

    async def boom(**kwargs):
        raise srv.RoutingError("bad router")

    fc.acompletion = boom
    r = c.post("/v1/chat/completions", json=_payload())
    assert r.status_code == 400
