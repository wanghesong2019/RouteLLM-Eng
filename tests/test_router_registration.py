"""注册与集成测试：remote_bert 可被 Controller 使用。

验证链路：Controller → RemoteBERTRouter → HTTP → 推理服务
"""
from unittest.mock import patch

import pytest


class TestRouterRegistration:
    def test_remote_bert_in_router_cls(self):
        from routellm.routers.routers import ROUTER_CLS

        assert "remote_bert" in ROUTER_CLS
        assert ROUTER_CLS["remote_bert"].__name__ == "RemoteBERTRouter"

    def test_upstream_routers_still_registered(self):
        """不应影响上游已有的路由器注册。"""
        from routellm.routers.routers import ROUTER_CLS

        for name in ("random", "mf", "causal_llm", "bert", "sw_ranking"):
            assert name in ROUTER_CLS

    def test_name_to_cls_roundtrip(self):
        from routellm.routers.routers import NAME_TO_CLS, ROUTER_CLS

        assert NAME_TO_CLS[ROUTER_CLS["remote_bert"]] == "remote_bert"


class TestControllerIntegration:
    """Controller 通过 config 构造 RemoteBERTRouter 并完成路由。"""

    def _fake_score(self, payload):
        n = len(payload["prompts"])
        # 简单规则：含 "hard" 的 prompt 高分（送强模型）
        results = [
            {"win_rate": 0.9 if "hard" in p else 0.1} for p in payload["prompts"]
        ]
        return {"results": results, "model_type": "bert", "count": n, "elapsed_ms": 1.0}

    def test_controller_routes_with_remote_bert(self):
        from routellm.controller import Controller

        with patch(
            "routellm.routers.remote.RemoteBERTRouter._post_score",
            side_effect=self._fake_score,
        ):
            c = Controller(
                routers=["remote_bert"],
                strong_model="strong-model",
                weak_model="weak-model",
                config={"remote_bert": {"base_url": "http://127.0.0.1:6070"}},
            )

            assert c.route("this is hard", "remote_bert", 0.5) == "strong-model"
            assert c.route("this is easy", "remote_bert", 0.5) == "weak-model"

    def test_model_name_parsing(self):
        """model="router-remote_bert-0.5" 形式应可解析。"""
        from routellm.controller import Controller

        with patch(
            "routellm.routers.remote.RemoteBERTRouter._post_score",
            side_effect=self._fake_score,
        ):
            c = Controller(
                routers=["remote_bert"],
                strong_model="strong-model",
                weak_model="weak-model",
                config={"remote_bert": {"base_url": "http://127.0.0.1:6070"}},
            )
            router, threshold = c._parse_model_name("router-remote_bert-0.5")
            assert router == "remote_bert"
            assert threshold == 0.5

    def test_model_counts_recorded_via_completion_path(self):
        """计数只在 completion 路径记录（上游行为：route() 不记计数）。"""
        from routellm.controller import Controller

        with patch(
            "routellm.routers.remote.RemoteBERTRouter._post_score",
            side_effect=self._fake_score,
        ):
            c = Controller(
                routers=["remote_bert"],
                strong_model="strong-model",
                weak_model="weak-model",
                config={"remote_bert": {"base_url": "http://127.0.0.1:6070"}},
            )
            msgs_hard = [{"role": "user", "content": "this is hard"}]
            msgs_easy = [{"role": "user", "content": "this is easy"}]
            assert c._get_routed_model_for_completion(msgs_hard, "remote_bert", 0.5) == "strong-model"
            assert c._get_routed_model_for_completion(msgs_easy, "remote_bert", 0.5) == "weak-model"
            assert c.model_counts["remote_bert"]["strong-model"] == 1
            assert c.model_counts["remote_bert"]["weak-model"] == 1

    def test_batch_calculate_win_rate(self):
        """评测路径：batch_calculate_win_rate 逐条调用 calculate_strong_win_rate。

        注意上游实现是 parallel_apply(calculate_strong_win_rate)，
        即每条 prompt 单独调一次，而非一次批量请求。
        """
        import pandas as pd

        from routellm.controller import Controller

        with patch(
            "routellm.routers.remote.RemoteBERTRouter._post_score",
            side_effect=self._fake_score,
        ):
            c = Controller(
                routers=["remote_bert"],
                strong_model="strong-model",
                weak_model="weak-model",
                config={"remote_bert": {"base_url": "http://127.0.0.1:6070"}},
            )
            series = pd.Series(["hard one", "easy one", "hard two"])
            out = c.batch_calculate_win_rate(series, "remote_bert")
            assert len(out) == 3
            assert out.iloc[0] == pytest.approx(0.9)
            assert out.iloc[1] == pytest.approx(0.1)
