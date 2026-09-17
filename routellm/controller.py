from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS


# ---------------------------------------------------------------------------
# 路由元信息（供监控体系读取）
#
# 背景：网关的监控需要「路由置信度 win_rate」与「路由计算延迟」，但
# _get_routed_model_for_completion 只返回模型名，把这两项丢掉了 ——
# 导致 Dashboard 的 win_rate 显示 N/A、routing_latency_ms 恒为 0
# （端到端验证时发现）。
#
# 用 ContextVar 而非实例属性：Controller 可能被多请求并发使用，
# 实例属性会互相覆盖导致指标串台。ContextVar 按协程隔离，天然安全。
# ---------------------------------------------------------------------------
@dataclass
class RoutingInfo:
    """一次路由决策的元信息。"""

    router: str
    threshold: float
    win_rate: float
    routed_model: str          # "strong" / "weak"
    latency_ms: float


_ROUTING_INFO: ContextVar = ContextVar("routellm_routing_info", default=None)


def _set_routing_info(info: RoutingInfo) -> None:
    """记录最近一次路由元信息（覆盖式）。"""
    _ROUTING_INFO.set(info)


def get_routing_info() -> Optional[RoutingInfo]:
    """读取本上下文最近一次路由元信息；无则 None。"""
    return _ROUTING_INFO.get()


# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    @property
    def last_routing_info(self) -> Optional[RoutingInfo]:
        """本上下文最近一次路由决策的元信息（供监控读取）。

        实际存取在模块级 ContextVar（见 RoutingInfo 的说明），
        这里只是给使用方一个直观的入口。
        """
        return get_routing_info()

    # ------------------------------------------------------------------
    # 运行时配置热更新（方案文档 4.8）
    #
    # 改造前：model_pair / api_base / api_key 在 __init__ 时固化。
    # 改造后：注入 config_store 时**按请求现取** —— 改配置无需重启即生效。
    #
    # 为何读操作无需加锁：RuntimeConfigStore 持不可变配置对象，update()
    # 是"整体换对象"而非就地修改，故读取方拿到的一定是完整一致的快照。
    # ------------------------------------------------------------------

    def live_config(self):
        """取当前生效的配置（热更新后即为新值）。

        未注入 config_store 时，用构造参数构造等价对象返回（向后兼容）。
        """
        if self.config_store is not None:
            return self.config_store.load()

        from routellm.config_runtime.store import RuntimeConfig

        return RuntimeConfig(
            strong_model=self.model_pair.strong,
            weak_model=self.model_pair.weak,
            api_base=self.api_base or "",
            api_key=self.api_key or "",
        )

    def live_model_pair(self):
        """取当前生效的强弱模型对（供路由决策与上报使用）。"""
        cfg = self.live_config()
        return ModelPair(strong=cfg.strong_model, weak=cfg.weak_model)

    def downstream_kwargs(self, model: str) -> dict:
        """构造下游 litellm 调用参数（base_url / api_key 按当前配置现取）。"""
        cfg = self.live_config()
        return {
            "model": model,
            "api_base": cfg.api_base or None,
            "api_key": cfg.api_key or None,
        }

    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
        config_store: Optional[Any] = None,
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        # 运行时配置存储（可选）。注入后，下游 LLM 的 base_url / api_key /
        # 模型名将**按请求现取**，从而支持不重启热更新（方案文档 4.8）。
        # 未注入时回落到构造参数，行为与改造前一致（向后兼容）。
        self.config_store = config_store

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        import time as _time

        prompt = messages[-1]["content"]
        # 用 live 配置取强弱模型（支持运行时热更新模型名）
        live_pair = self.live_model_pair()

        t0 = _time.perf_counter()
        router_instance = self.routers[router]
        # 显式算 win_rate 而非直接调 route() —— 后者只返回模型名，
        # 会把置信度丢掉（监控需要它）
        try:
            win_rate = float(router_instance.calculate_strong_win_rate(prompt))
            routed_model = (
                live_pair.strong if win_rate >= threshold else live_pair.weak
            )
        except Exception:  # noqa: BLE001
            # 路由失败时回落到原接口（保持既有容错行为）
            routed_model = router_instance.route(prompt, threshold, live_pair)
            win_rate = 0.0
        latency_ms = (_time.perf_counter() - t0) * 1000

        self.model_counts[router][routed_model] += 1

        # 记录元信息供监控体系读取（见 RoutingInfo 的说明）
        _set_routing_info(
            RoutingInfo(
                router=router,
                threshold=float(threshold),
                win_rate=win_rate,
                routed_model=(
                    "strong" if routed_model == live_pair.strong else "weak"
                ),
                latency_ms=round(latency_ms, 3),
            )
        )

        return routed_model

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)

        return self.routers[router].route(prompt, threshold, self.model_pair)

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        # 用 live 配置构造下游参数 —— 支持运行时热更新（方案文档 4.8）
        kw = self.downstream_kwargs(kwargs["model"])
        kwargs.pop("model")
        return completion(**kw, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        # 用 live 配置构造下游参数 —— 支持运行时热更新（方案文档 4.8）
        kw = self.downstream_kwargs(kwargs["model"])
        kwargs.pop("model")
        return await acompletion(**kw, **kwargs)
