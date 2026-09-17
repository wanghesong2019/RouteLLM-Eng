"""FastAPI 中间件：请求级指标采集。

设计依据：方案文档 4.2.4。改造前只有 `logging.info(model_counts)`（无结构化、
重启丢失）。本中间件在每次 `/v1/chat/completions` 请求后采集结构化指标，
写入 SQLite 并更新 Prometheus 计数器。

为何只拦 /v1/chat/completions：
    /health、/v1/models、/dashboard 等是运维/展示端点，不产生路由成本，
    记录它们会污染成本与延迟统计。

写入方式：异步 fire-and-forget（`asyncio.create_task`）——
不阻塞响应返回。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable, Optional

from routellm.monitoring import prometheus
from routellm.monitoring.metrics import RequestMetrics, estimate_cost
from routellm.monitoring.store import MetricsStore

logger = logging.getLogger(__name__)

TRACKED_PATH = "/v1/chat/completions"


class MetricsMiddleware:
    """采集 /v1/chat/completions 的路由与成本指标。

    指标来源分两部分：
        1. 中间件自身可测的：总延迟、状态码
        2. 由请求处理函数写入 contextvar 的：路由决策、win_rate、
           token 数、成本、缓存命中

    处理函数若未写入（如异常路径），相关字段置 None/0 而非报错。

    为何从 contextvar 读取而非 request.state：
        网关的处理函数签名是 pydantic 模型（非 Starlette Request），
        无法挂 state。ContextVar 天然按协程隔离，并发安全。
    """

    def __init__(
        self,
        app: Callable,
        store: Optional[MetricsStore] = None,
        ctx_var: Optional[object] = None,
    ):
        # 注意参数顺序：Starlette 的 add_middleware 以 cls(app, *args, **kwargs)
        # 构造中间件 —— app 必须是第一个位置参数，其余走 kwargs。
        # 实测踩过：写成 (store, app) 会报
        # "got multiple values for argument 'store'"。
        self.app = app
        self.store = store
        self.ctx_var = ctx_var

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != TRACKED_PATH:
            return await self.app(scope, receive, send)

        start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:  # noqa: BLE001
            ctx = dict(self._ctx() or {})
            ctx["error_message"] = f"{type(e).__name__}: {e}"
            self._set_ctx(ctx)
            raise
        finally:
            total_ms = (time.perf_counter() - start) * 1000
            try:
                self._record(self._ctx() or {}, request_id, total_ms, status_code)
            except Exception as e:  # noqa: BLE001
                logger.warning("指标采集失败（不影响请求）: %s", e)

    def _ctx(self) -> dict:
        if self.ctx_var is None:
            return {}
        v = self.ctx_var.get()
        return v if isinstance(v, dict) else {}

    def _set_ctx(self, ctx: dict) -> None:
        if self.ctx_var is not None:
            self.ctx_var.set(ctx)

    def _record(self, ctx: dict, request_id: str, total_ms: float, status_code: int) -> None:
        routed_model = ctx.get("routed_model") or "unknown"
        pt = int(ctx.get("prompt_tokens") or 0)
        ct = int(ctx.get("completion_tokens") or 0)
        cost = ctx.get("estimated_cost")
        if cost is None:
            cost = estimate_cost(routed_model if routed_model in ("strong", "weak") else "weak", pt, ct)

        status = "success" if 200 <= status_code < 400 else "error"

        m = RequestMetrics(
            request_id=request_id,
            timestamp=time.time(),
            prompt_hash=ctx.get("prompt_hash") or "",
            router_name=ctx.get("router_name") or "",
            threshold=float(ctx.get("threshold") or 0.0),
            win_rate=float(ctx.get("win_rate") or 0.0),
            routed_model=routed_model,
            routing_latency_ms=float(ctx.get("routing_latency_ms") or 0.0),
            llm_latency_ms=float(ctx.get("llm_latency_ms") or 0.0),
            total_latency_ms=total_ms,
            prompt_tokens=pt,
            completion_tokens=ct,
            estimated_cost=float(cost),
            cache_hit=bool(ctx.get("cache_hit")),
            status=status,
            error_message=ctx.get("error_message"),
        )

        # 异步写入，不阻塞响应
        if self.store is not None:
            asyncio.create_task(self.store.save(m))

        # Prometheus
        prometheus.inc_counter("routellm_requests_total", {"model": m.routed_model})
        prometheus.observe_histogram("routellm_request_latency_ms", m.total_latency_ms)
        prometheus.observe_histogram("routellm_routing_latency_ms", m.routing_latency_ms)
        if m.status == "error":
            prometheus.inc_counter("routellm_errors_total")
        if m.cache_hit:
            prometheus.inc_counter("routellm_cache_hits_total")
