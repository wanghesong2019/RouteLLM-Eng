"""降级策略链（方案文档 4.3）。

四条降级路径
------------
    ① 路由器异常 → 直接走弱模型
    ② 强模型调用失败 → 降级到弱模型
    ③ 弱模型也失败 → 查缓存返回历史响应（标记 downgraded）
    ④ 缓存也无 → 抛 FallbackExhaustedError（上层转 503 + Retry-After: 30）

设计要点
--------
1. **降级必须可观测**：任何非原始路由结果的响应都应带
   `X-RouteLLM-Downgraded: true`，否则客户端会把降级结果误认为正常路由。
   本模块用 `FallbackResult.downgraded` 表达，由 server 层转成响应头。
2. **熔断开路时不再尝试**：若强模型熔断器已开路，直接走弱模型 ——
   否则每次请求都要白等一次超时，把上游故障放大成网关侧延迟。
3. **缓存兜底复用既有 key 格式**：`routellm:res:<hash>`（见 cache/keys.py），
   与 win_rate 结果缓存共用一套 key 约定，避免第二套缓存语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from routellm.resilience.circuit_breaker import CircuitBreaker, CircuitState
from routellm.resilience.retry import with_retry


class FallbackExhaustedError(Exception):
    """降级链全部耗尽（强弱模型与缓存都不可用）。

    由 server 层转成 HTTP 503 + `Retry-After`，让客户端知道「稍后再试」
    而不是「请求有错」。
    """

    def __init__(self, message: str = "All upstream models unavailable",
                 retry_after: int = 30) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class FallbackResult:
    """降级链的执行结果。"""

    value: Any
    tier: str            # "strong" / "weak"
    source: str          # "strong" / "weak" / "cache"
    downgraded: bool     # True 表示响应不是原始路由结果


class ResilientCaller:
    """按降级链调用强弱模型。

    参数
        cache:           可选缓存（接口 get(key)/set(key,value)）。
                         通常传 MultiTierCache 实例。
        strong_breaker:  强模型熔断器（可选）
        weak_breaker:    弱模型熔断器（可选）
        router_breaker:  路由器熔断器（可选）
    """

    def __init__(
        self,
        cache: Optional[Any] = None,
        strong_breaker: Optional[CircuitBreaker] = None,
        weak_breaker: Optional[CircuitBreaker] = None,
        router_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.cache = cache
        self.strong_breaker = strong_breaker
        self.weak_breaker = weak_breaker
        self.router_breaker = router_breaker

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _breaker_allows(breaker: Optional[CircuitBreaker]) -> bool:
        """无熔断器视为允许；有则询问。"""
        if breaker is None:
            return True
        return breaker.allow_request()

    @staticmethod
    def _record(breaker: Optional[CircuitBreaker], ok: bool) -> None:
        if breaker is None:
            return
        if ok:
            breaker.record_success()
        else:
            breaker.record_failure()

    def _cache_get(self, key: Optional[str]) -> Optional[Any]:
        if self.cache is None or not key:
            return None
        try:
            return self.cache.get(key)
        except Exception:  # noqa: BLE001
            return None  # 缓存故障不应让降级链断掉

    def _cache_set(self, key: Optional[str], value: Any) -> None:
        if self.cache is None or not key:
            return
        try:
            self.cache.set(key, value)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    async def call_with_fallback(
        self,
        *,
        strong_fn: Optional[Callable[[], Awaitable[Any]]],
        weak_fn: Callable[[], Awaitable[Any]],
        cache_key: Optional[str] = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        jitter: float = 0.1,
        router_ok: bool = True,
        retry_enabled: bool = True,
    ) -> FallbackResult:
        """按降级链执行：强 → 弱 → 缓存 → 抛错。

        Args:
            strong_fn: 强模型调用（None 表示本次不使用强模型，如路由判定走弱）。
            weak_fn: 弱模型调用（必需 —— 它是降级链的最后一道模型防线）。
            cache_key: 缓存 key，用于兜底与成功回填。
            router_ok: 路由器是否正常。False 时跳过强模型直接走弱（降级链①）。

        返回值总是 FallbackResult；只有降级链完全耗尽才抛
        FallbackExhaustedError。
        """
        from routellm.resilience.retry import is_retryable  # 局部导入避免循环

        async def _run(fn, breaker):
            """在熔断器 + 重试保护下执行一次调用。"""
            if retry_enabled:
                return await with_retry(
                    fn, max_attempts=max_attempts, base_delay=base_delay,
                    max_delay=max_delay, jitter=jitter,
                )
            return await fn()

        # ---- 强模型（降级链①：路由器异常则跳过）----
        strong_skipped = (not router_ok) or strong_fn is None
        if not strong_skipped:
            if not self._breaker_allows(self.strong_breaker):
                # 熔断开路：不白等一次超时，直接降级
                strong_skipped = True
            else:
                try:
                    value = await _run(strong_fn, self.strong_breaker)
                    self._record(self.strong_breaker, True)
                    self._cache_set(cache_key, value)
                    return FallbackResult(value=value, tier="strong",
                                          source="strong", downgraded=False)
                except Exception as exc:  # noqa: BLE001
                    self._record(self.strong_breaker, False)
                    # 客户端错误（参数/鉴权等）不可通过换模型修复：
                    # 降级到弱模型只会把同一个错误再犯一次，最终还会把本该
                    # 透传的 4xx 伪装成 503，误导客户端去重试。
                    # 因此直接抛出，不进降级链。
                    if not is_retryable(exc):
                        raise

        # ---- 弱模型（降级链②）----
        if self._breaker_allows(self.weak_breaker):
            try:
                value = await _run(weak_fn, self.weak_breaker)
                self._record(self.weak_breaker, True)
                self._cache_set(cache_key, value)
                return FallbackResult(value=value, tier="weak",
                                      source="weak", downgraded=True)
            except Exception as exc:  # noqa: BLE001
                self._record(self.weak_breaker, False)
                if not is_retryable(exc):
                    raise
        else:
            self._record(self.weak_breaker, False)

        # ---- 缓存兜底（降级链③）----
        cached = self._cache_get(cache_key)
        if cached is not None:
            return FallbackResult(value=cached, tier="weak",
                                  source="cache", downgraded=True)

        # ---- 全部耗尽（降级链④）----
        raise FallbackExhaustedError(retry_after=30)
