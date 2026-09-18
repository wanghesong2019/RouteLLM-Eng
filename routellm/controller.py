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
        """取当前生效的强弱模型对（供路由决策与上报使用）。

        含**单模型兜底**：某一档模型名为空时并入已配的那一档，
        因此这里返回的两个名字都不会是空串（除非两侧都空 → 此时抛错，
        由调用方决定如何处理，不会带着空模型名去调下游）。

        未注入 config_store 时用构造参数（向后兼容，行为不变）。
        """
        cfg = self.live_config()
        if self.config_store is not None:
            pair = self.config_store.effective_model_pair()
            return ModelPair(strong=pair.strong, weak=pair.weak)
        return ModelPair(strong=cfg.strong_model, weak=cfg.weak_model)

    def downstream_kwargs(self, model: str, tier: Optional[str] = None) -> dict:
        """构造下游 litellm 调用参数（base_url / api_key 按当前配置现取）。

        Args:
            model: **原始模型名**（如 `deepseek-ai/DeepSeek-V4-Pro`）。
                前缀由本方法统一拼接 —— 用户配置里不写前缀（方案 a）。
            tier: "strong" / "weak"，决定用哪一侧的 base_url / api_key；
                None 表示用顶层全局配置（向后兼容）。

        Returns:
            含 model / api_base / api_key 的 dict，可直接 **kwargs 给 litellm。
        """
        cfg = self.live_config()
        if self.config_store is not None:
            base = self.config_store.effective_api_base(tier)
            key = self.config_store.effective_api_key(tier)
        else:
            base, key = cfg.api_base, cfg.api_key
        return {
            "model": self._qualify_model(model),
            "api_base": base or None,
            "api_key": key or None,
        }

    @staticmethod
    def _qualify_model(model: str) -> str:
        """给原始模型名补 provider 前缀（litellm 靠前缀选择适配器）。

        只在此处拼一次 —— 配置层与前端都存/显示原始名。
        已带前缀的名字原样返回（避免 `openai/openai/...`）。

        注意：前缀是**必需**的，直接去掉会让 litellm 抛 BadRequestError。
        所以拼接必须发生在调用前的最后一步，且只此一处。
        """
        from routellm.config_runtime.store import DEFAULT_PROVIDER_PREFIX

        m = (model or "").strip()
        if not m:
            return m
        # 已知 provider 前缀（litellm 支持的常见值）—— 用户若显式带了就尊重原样。
        # 用白名单而非"含斜杠即视为前缀"：模型名本身也常含斜杠
        # （如 deepseek-ai/DeepSeek-V4-Pro），后者会误判。
        known = {
            "openai", "azure", "anthropic", "bedrock", "vertex_ai", "gemini",
            "cohere", "mistral", "ollama", "together_ai", "deepseek",
            "groq", "xai", "openrouter", "siliconflow", "hosted_vllm",
        }
        head = m.split("/")[0].lower() if "/" in m else ""
        if head in known:
            return m
        return f"{DEFAULT_PROVIDER_PREFIX}/{m}"


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
        resilience_enabled: bool = False,
        resilience_cache: Optional[Any] = None,
        resilience_max_attempts: int = 3,
        resilience_failure_threshold: int = 5,
        resilience_recovery_timeout: float = 60.0,
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

        # ---- 容错与 Resilience（方案文档 4.3）----
        # 默认关闭：不改变既有部署行为。开启后，下游调用被 重试 + 熔断 + 降级
        # 包裹（强 → 弱 → 缓存 → 503）。
        self.resilience_enabled = resilience_enabled
        self.resilience_cache = resilience_cache
        self.resilience_max_attempts = resilience_max_attempts
        self.resilience_failure_threshold = resilience_failure_threshold
        self.resilience_recovery_timeout = resilience_recovery_timeout
        # 熔断器按 "router|tier" 分片 —— 不同路由器、强弱两侧互不影响
        # （方案文档 4.3 步骤 5：按路由器名和模型名分别实例化）
        self._breakers: dict[str, Any] = {}
        # 最近一次调用是否降级（供 server 层写 X-RouteLLM-Downgraded 头）。
        # 用 ContextVar 语义由 server 层保证隔离；此处存实例属性 + 逐请求重置。
        self.last_downgraded = False

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

    def _get_breaker(self, router: str, tier: str):
        """取（或创建）该「路由器 × 档位」分片的熔断器。

        分片理由：不同路由器的健康度不同；强模型挂了不代表弱模型也挂 ——
        共用一个熔断器会让一侧的故障错误地掐断另一侧的正常流量。
        """
        from routellm.resilience import CircuitBreaker

        key = f"{router}|{tier}"
        cb = self._breakers.get(key)
        if cb is None:
            cb = CircuitBreaker(
                failure_threshold=self.resilience_failure_threshold,
                recovery_timeout=self.resilience_recovery_timeout,
            )
            self._breakers[key] = cb
        return cb

    def _cache_key_for_messages(self, messages: list) -> str:
        """降级链③的缓存兜底 key。

        复用 cache/keys.py 的 result_key 约定（routellm:res:<hash>），
        避免引入第二套 key 语义。
        """
        import json as _json

        from routellm.cache.keys import result_key

        try:
            payload = _json.dumps(messages, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = str(messages)
        return result_key(payload)

    async def _call_downstream(
        self, *, router: str, routed_model: str, tier: str, messages: list, kwargs: dict
    ):
        """在容错保护下调用下游 LLM（方案文档 4.3 主流程）。

        降级链：强 → 弱 → 缓存 → FallbackExhaustedError。
        未启用容错时走裸调用（保持改造前行为）。
        """
        live_pair = self.live_model_pair()
        strong_model = live_pair.strong
        weak_model = live_pair.weak
        cache_key = self._cache_key_for_messages(messages)

        if not self.resilience_enabled:
            kw = self.downstream_kwargs(routed_model, tier)
            return await acompletion(**kw, **kwargs)

        from routellm.resilience import FallbackExhaustedError, ResilientCaller

        caller = ResilientCaller(
            cache=self.resilience_cache,
            strong_breaker=self._get_breaker(router, "strong"),
            weak_breaker=self._get_breaker(router, "weak"),
        )

        async def _invoke(model_name: str, tier_name: str):
            kw = self.downstream_kwargs(model_name, tier_name)
            return await acompletion(**kw, **kwargs)

        # 路由实际选中的是强模型时，才把「强」作为首选项；
        # 路由判定走弱时 strong_fn=None，直接走弱（不浪费一次强模型调用）。
        use_strong_first = (routed_model == strong_model) and (strong_model != weak_model)

        try:
            if use_strong_first:
                res = await caller.call_with_fallback(
                    strong_fn=lambda: _invoke(strong_model, "strong"),
                    weak_fn=lambda: _invoke(weak_model, "weak"),
                    cache_key=cache_key,
                    max_attempts=self.resilience_max_attempts,
                )
            else:
                res = await caller.call_with_fallback(
                    strong_fn=None,
                    weak_fn=lambda: _invoke(weak_model, "weak"),
                    cache_key=cache_key,
                    max_attempts=self.resilience_max_attempts,
                )
            self.last_downgraded = bool(res.downgraded)
            return res.value
        except FallbackExhaustedError:
            self.last_downgraded = True
            raise

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
            # 路由失败时回落到原接口（保持既有容错行为）。
            # 注意：若路由器服务**整体不可达**（如 BERT 6070 挂掉），
            # 这次回落同样会失败 —— 此时：
            #   - 启用容错：直接判定走弱模型（降级链①），让下游降级逻辑
            #     正常接管，而不是把路由层异常抛给客户端
            #   - 未启用容错：保持改造前行为，原样抛出
            try:
                routed_model = router_instance.route(prompt, threshold, live_pair)
            except Exception:  # noqa: BLE001
                if not self.resilience_enabled:
                    raise
                routed_model = live_pair.weak
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
        self.last_downgraded = False  # 逐请求重置降级标记

        messages = kwargs["messages"]
        routed_model = self._get_routed_model_for_completion(
            messages, router, threshold
        )
        live_pair = self.live_model_pair()
        # 判定路由落在哪一侧 —— 决定降级链的首选项与凭据档位
        tier = "strong" if routed_model == live_pair.strong else "weak"

        # 构造下游参数时不把 model 传给 litellm（由 _call_downstream 决定）
        call_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        return await self._call_downstream(
            router=router, routed_model=routed_model, tier=tier,
            messages=messages, kwargs=call_kwargs,
        )

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        """同步调用。容错启用时仍走异步降级链（内部同步等待）。"""
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)

        if not self.resilience_enabled:
            self.last_downgraded = False
            routed_model = self._get_routed_model_for_completion(
                kwargs["messages"], router, threshold
            )
            kw = self.downstream_kwargs(routed_model)
            kwargs.pop("model", None)
            return completion(**kw, **kwargs)

        # 容错路径：复用异步实现，避免两套降级逻辑漂移
        import asyncio as _asyncio

        coro = self.acompletion(router=router, threshold=threshold, **kwargs)
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            return _asyncio.run(coro)
        # 已在事件循环中：不能 asyncio.run，交给调用方用 acreate
        coro.close()
        raise RoutingError(
            "completion() cannot be used with resilience enabled inside a "
            "running event loop; use acompletion() instead."
        )
