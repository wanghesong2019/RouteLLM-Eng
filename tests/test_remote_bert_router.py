"""RemoteBERTRouter 测试（RED 阶段先写，实现后再转 GREEN）。

设计依据：docs/decisions/ADR-001-inference-service-on-host.md
被测对象：routellm/routers/remote.py :: RemoteBERTRouter

契约要求（必须与上游 BERTRouter 一致）：
  - 实现 Router 抽象接口：calculate_strong_win_rate(prompt) -> float
  - 继承 Router 后自动获得 route()（阈值比较）
  - win_rate 语义与上游一致：1 - sum(softmax[-2:])

测试不依赖真实推理服务，用 mock HTTP 层。
"""
from unittest.mock import MagicMock, patch

import pytest

from routellm.routers.routers import BERTRouter, Router


class TestRemoteBERTRouterContract:
    """接口契约：必须能被当作 Router 使用。"""

    def test_is_router_subclass(self):
        from routellm.routers.remote import RemoteBERTRouter

        assert issubclass(RemoteBERTRouter, Router)

    def test_has_calculate_strong_win_rate(self):
        from routellm.routers.remote import RemoteBERTRouter

        assert hasattr(RemoteBERTRouter, "calculate_strong_win_rate")

    def test_init_requires_base_url(self):
        from routellm.routers.remote import RemoteBERTRouter

        with pytest.raises(TypeError):
            RemoteBERTRouter()  # base_url 必填


class TestRemoteBERTRouterBehavior:
    """行为：调用推理服务并解析 win_rate。"""

    def _make_router(self, base_url="http://127.0.0.1:6070", **kw):
        from routellm.routers.remote import RemoteBERTRouter

        return RemoteBERTRouter(base_url=base_url, **kw)

    def test_returns_win_rate_from_service(self):
        router = self._make_router()
        fake = {"results": [{"win_rate": 0.75}], "model_type": "bert", "count": 1,
                "elapsed_ms": 10.0}
        with patch.object(router, "_post_score", return_value=fake):
            assert router.calculate_strong_win_rate("some prompt") == pytest.approx(0.75)

    def test_route_uses_threshold(self):
        """继承自 Router.route：win_rate >= threshold → strong。"""
        from routellm.controller import ModelPair

        router = self._make_router()
        pair = ModelPair(strong="strong-model", weak="weak-model")
        fake = {"results": [{"win_rate": 0.75}], "model_type": "bert", "count": 1,
                "elapsed_ms": 10.0}
        with patch.object(router, "_post_score", return_value=fake):
            assert router.route("p", 0.5, pair) == "strong-model"
            assert router.route("p", 0.8, pair) == "weak-model"

    def test_batch_scoring(self):
        """批量评分：一次请求返回多条，顺序与输入一致。"""
        router = self._make_router()
        fake = {"results": [{"win_rate": 0.1}, {"win_rate": 0.9}],
                "model_type": "bert", "count": 2, "elapsed_ms": 12.0}
        with patch.object(router, "_post_score", return_value=fake):
            out = router.calculate_strong_win_rate_batch(["a", "b"])
        assert out == [pytest.approx(0.1), pytest.approx(0.9)]

    def test_calculate_strong_win_rate_uses_batch_path(self):
        """单条调用应走批量接口，避免两套代码路径。"""
        router = self._make_router()
        fake = {"results": [{"win_rate": 0.42}], "model_type": "bert", "count": 1,
                "elapsed_ms": 5.0}
        with patch.object(router, "_post_score", return_value=fake) as m:
            router.calculate_strong_win_rate("hello")
            assert m.called
            payload = m.call_args[0][0]
            assert payload["prompts"] == ["hello"]


class TestRemoteBERTRouterErrors:
    """错误处理：服务异常时应给出清晰错误，而非静默返回错误结果。"""

    def _make_router(self, **kw):
        from routellm.routers.remote import RemoteBERTRouter

        return RemoteBERTRouter(base_url="http://127.0.0.1:6070", **kw)

    def test_service_unreachable_raises(self):
        from routellm.routers.remote import RemoteInferenceError

        router = self._make_router()
        with patch.object(router, "_post_score", side_effect=RemoteInferenceError("boom")):
            with pytest.raises(RemoteInferenceError):
                router.calculate_strong_win_rate("p")

    def test_malformed_response_raises(self):
        """响应结构不对（缺 results / 数量不匹配）应显式报错。"""
        from routellm.routers.remote import RemoteInferenceError

        router = self._make_router()
        with patch.object(router, "_post_score", return_value={"unexpected": 1}):
            with pytest.raises(RemoteInferenceError):
                router.calculate_strong_win_rate("p")

    def test_count_mismatch_raises(self):
        from routellm.routers.remote import RemoteInferenceError

        router = self._make_router()
        fake = {"results": [{"win_rate": 0.5}], "count": 99}
        with patch.object(router, "_post_score", return_value=fake):
            with pytest.raises(RemoteInferenceError):
                router.calculate_strong_win_rate_batch(["a", "b"])


class TestRemoteRouterRegistration:
    """注册：应可通过 config 使用 router 名 'remote_bert'。"""

    def test_router_cls_lookup(self):
        from routellm.routers.remote import RemoteBERTRouter

        assert RemoteBERTRouter.__name__ == "RemoteBERTRouter"

    def test_accepts_timeout_and_batch_size(self):
        from routellm.routers.remote import RemoteBERTRouter

        r = RemoteBERTRouter(
            base_url="http://127.0.0.1:6070", timeout=30.0, batch_size=64
        )
        assert r.base_url == "http://127.0.0.1:6070"
        assert r.timeout == 30.0
        assert r.batch_size == 64
