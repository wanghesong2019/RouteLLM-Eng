"""remote_sw_ranking 路由的单元测试（TDD RED 阶段）。

背景：
    sw_ranking 已服务化为 host 侧独立服务（services/sw_ranking_server.py，
    端口 6071），接口与 BERT 服务（6070）的 /v1/score 对齐。
    现在需要在 RouteLLM 侧新增一个 remote_sw_ranking 路由，通过 HTTP 调用它。

设计要点：
    - remote_sw_ranking 与 remote_bert 接口形状相同，故复用同一套 HTTP 客户端
      代码（_post_score / batch 分片 / 错误处理），只换 base_url。
    - 但两者端口不同（6070 vs 6071），而 _inject_inference_url 当前只注入
      单一 ROUTELLM_INFERENCE_URL。需要支持按路由分别指定 URL：
        ROUTELLM_INFERENCE_URL          → 默认（remote_bert）
        ROUTELLM_SW_RANKING_INFERENCE_URL → remote_sw_ranking 专用（可选覆盖）
    - win_rate 语义：两者都是「应路由到强模型的程度」，可被 Controller 无缝替换。

运行：
    pytest tests/test_remote_sw_ranking.py -v
"""

import json

import pytest


# --------------------------------------------------------------- 路由注册


def test_remote_sw_ranking_registered():
    """ROUTER_CLS 中应注册 remote_sw_ranking。"""
    from routellm.routers.routers import ROUTER_CLS

    assert "remote_sw_ranking" in ROUTER_CLS, (
        f"未注册 remote_sw_ranking，当前有: {list(ROUTER_CLS)}"
    )


def test_router_is_subclass_of_router():
    """须实现 Router 抽象接口，才能被 Controller 使用。"""
    from routellm.routers.remote import RemoteSWRankingRouter
    from routellm.routers.routers import Router

    assert issubclass(RemoteSWRankingRouter, Router)


# --------------------------------------------------------------- HTTP 契约


def test_default_port_hint_is_6071():
    """默认 base_url 应指向 6071（sw_ranking 服务），而非 6070（bert）。"""
    from routellm.routers.remote import RemoteSWRankingRouter

    r = RemoteSWRankingRouter()
    assert "6071" in r.base_url, f"默认应指向 6071，实际 {r.base_url}"


def test_accepts_explicit_base_url():
    """应能显式指定 base_url。"""
    from routellm.routers.remote import RemoteSWRankingRouter

    r = RemoteSWRankingRouter(base_url="http://host.docker.internal:6071")
    assert r.base_url == "http://host.docker.internal:6071"


def test_posts_to_v1_score(monkeypatch):
    """须 POST 到 /v1/score，请求体含 prompts。"""
    from routellm.routers.remote import RemoteSWRankingRouter

    captured = {}

    def fake_post(self, payload):
        captured["payload"] = payload
        return {"results": [{"win_rate": 0.69}], "model_type": "sw_ranking",
                "count": 1, "elapsed_ms": 1.0}

    monkeypatch.setattr(RemoteSWRankingRouter, "_post_score", fake_post)

    r = RemoteSWRankingRouter(base_url="http://x:6071")
    wr = r.calculate_strong_win_rate("hello")
    assert wr == pytest.approx(0.69)
    assert captured["payload"]["prompts"] == ["hello"]


def test_batch_preserves_order_and_splits(monkeypatch):
    """批量应分片且保序。"""
    from routellm.routers.remote import RemoteSWRankingRouter

    calls = []

    def fake_post(self, payload):
        calls.append(list(payload["prompts"]))
        return {
            "results": [{"win_rate": float(i)} for i in range(len(payload["prompts"]))],
            "model_type": "sw_ranking",
            "count": len(payload["prompts"]),
            "elapsed_ms": 1.0,
        }

    monkeypatch.setattr(RemoteSWRankingRouter, "_post_score", fake_post)

    r = RemoteSWRankingRouter(base_url="http://x:6071", batch_size=2)
    wrs = r.calculate_strong_win_rate_batch(["a", "b", "c", "d", "e"])
    assert len(wrs) == 5
    assert len(calls) == 3, f"应分成 3 片（2+2+1），实际 {len(calls)}"


def test_error_handling_on_http_failure():
    """HTTP 失败须抛 RemoteInferenceError（与其他 remote 路由一致）。"""
    from routellm.routers.remote import RemoteInferenceError, RemoteSWRankingRouter

    r = RemoteSWRankingRouter(base_url="http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(RemoteInferenceError):
        r.calculate_strong_win_rate("hi")


# --------------------------------------------------------------- URL 注入


def test_inject_url_supports_per_router_override():
    """_inject_inference_url 须支持 remote_sw_ranking 用独立 URL。

    因为 bert(6070) 与 sw_ranking(6071) 是两个服务、两个端口。
    """
    from routellm.openai_server import _inject_inference_url

    class FakeSettings:
        inference_url = "http://host.docker.internal:6070"
        sw_ranking_inference_url = "http://host.docker.internal:6071"
        routers = ["remote_bert", "remote_sw_ranking"]

    cfg = _inject_inference_url({}, FakeSettings())
    assert cfg["remote_bert"]["base_url"] == "http://host.docker.internal:6070"
    assert cfg["remote_sw_ranking"]["base_url"] == "http://host.docker.internal:6071"


def test_inject_url_falls_back_to_default():
    """未配置 sw_ranking 专用 URL 时，回落默认（不报错）。"""
    from routellm.openai_server import _inject_inference_url

    class FakeSettings:
        inference_url = "http://host.docker.internal:6070"
        sw_ranking_inference_url = None
        routers = ["remote_bert", "remote_sw_ranking"]

    cfg = _inject_inference_url({}, FakeSettings())
    assert cfg["remote_bert"]["base_url"] == "http://host.docker.internal:6070"
    assert cfg["remote_sw_ranking"]["base_url"] == "http://host.docker.internal:6070"


def test_inject_url_respects_existing_config():
    """配置文件里已显式指定的 base_url 不应被覆盖。"""
    from routellm.openai_server import _inject_inference_url

    class FakeSettings:
        inference_url = "http://default:6070"
        sw_ranking_inference_url = "http://sw:6071"
        routers = ["remote_sw_ranking"]

    cfg = _inject_inference_url(
        {"remote_sw_ranking": {"base_url": "http://explicit:9999"}}, FakeSettings()
    )
    assert cfg["remote_sw_ranking"]["base_url"] == "http://explicit:9999"
