"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import contextvars
import logging
import os
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
import yaml
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from routellm.config import DEFAULT_ROUTERS, ConfigError, Settings
from routellm.controller import Controller, RoutingError
from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None
SETTINGS: Optional[Settings] = None

count = defaultdict(lambda: defaultdict(int))


@asynccontextmanager
async def lifespan(app):
    global CONTROLLER, SETTINGS

    SETTINGS = Settings.from_env()
    # 启动时校验，fail fast —— 避免配置错误拖到运行时变成 500
    try:
        SETTINGS.validate()
    except ConfigError as e:
        logging.error("配置校验失败，服务无法启动: %s", e)
        raise

    logging.info("配置: %s", SETTINGS.summary())
    if SETTINGS.inference_url:
        logging.info("路由模型推理服务: %s", SETTINGS.inference_url)

    router_config = (
        yaml.safe_load(open(SETTINGS.config_path, "r"))
        if SETTINGS.config_path
        else None
    )
    # 若配置了远程推理服务，为 remote_* 路由器注入 base_url
    router_config = _inject_inference_url(router_config, SETTINGS)

    CONTROLLER = Controller(
        routers=SETTINGS.routers,
        config=router_config,
        strong_model=SETTINGS.strong_model,
        weak_model=SETTINGS.weak_model,
        api_base=SETTINGS.api_base,
        api_key=SETTINGS.api_key,
        progress_bar=True,
        config_store=_CONFIG_STORE,  # 注入后支持运行时热更新（方案文档 4.8）
    )
    yield
    CONTROLLER = None


def _inject_inference_url(config, settings):
    """把推理服务 URL 注入到 remote_* 路由器的配置中。

    默认注入 ROUTELLM_INFERENCE_URL（remote_bert 用，端口 6070）。
    remote_sw_ranking 指向独立的 sw_ranking 服务（端口 6071），
    可用 ROUTELLM_SW_RANKING_INFERENCE_URL 单独覆盖；未设置时回落默认。

    注：setdefault 保证配置文件里已显式指定的 base_url 不被覆盖。
    """
    if not settings.inference_url:
        return config
    cfg = dict(config) if config else {}

    # 路由名 → 专用 URL 环境变量（None 表示用默认）
    override = {
        "remote_sw_ranking": getattr(settings, "sw_ranking_inference_url", None),
    }

    for name in settings.routers:
        if name.startswith("remote_"):
            cfg.setdefault(name, {})
            url = override.get(name) or settings.inference_url
            cfg[name].setdefault("base_url", url)
    return cfg


# 指标采集上下文。用 ContextVar 而非全局 dict —— 网关处理并发请求时，
# 全局 dict 会被多个请求互相覆盖，导致指标串台。ContextVar 天然按协程隔离。
_METRICS_CTX: contextvars.ContextVar = contextvars.ContextVar("routellm_metrics", default=None)

# 运行时配置存储（方案文档 4.8 —— 配置热更新）。
# 在模块级创建（lifespan 之前），因为 lifespan 里构造 Controller 时要用到。
# 优先读 ROUTELLM_RUNTIME_CONFIG 指定的 JSON，回落环境变量。
try:
    from routellm.config_runtime.store import RuntimeConfigStore as _RuntimeConfigStore

    _CONFIG_STORE = _RuntimeConfigStore()
except Exception as _e:  # noqa: BLE001
    _CONFIG_STORE = None


app = fastapi.FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# API key 鉴权
#
# 网关对外暴露（0.0.0.0），必须鉴权 —— 否则任何人可白嫖下游 LLM 额度。
# 客户端用法与普通模型 API 服务一致：Authorization: Bearer <key>
#
# 未配置 ROUTELLM_GATEWAY_API_KEY 时不鉴权（内网/回环部署的向后兼容）。
# 白名单：/health、/dashboard、/metrics、/api/*（运维用途，不要求业务 key）。
# ---------------------------------------------------------------------------
try:
    from routellm.monitoring.auth import ApiKeyMiddleware as _ApiKeyMiddleware
    from routellm.monitoring.auth import is_auth_enabled as _auth_enabled

    app.add_middleware(_ApiKeyMiddleware)
    if _auth_enabled():
        logging.info("API key 鉴权已启用（/v1/* 需 Authorization: Bearer）")
    else:
        logging.warning(
            "API key 鉴权未启用（未设置 ROUTELLM_GATEWAY_API_KEY）——"
            "对外暴露时请务必配置"
        )
except Exception as _e:  # noqa: BLE001
    logging.warning("鉴权中间件初始化失败: %s", _e)

# ---------------------------------------------------------------------------
# 监控（方案文档 4.2，P0）
#
# 挂载自研 Dashboard + Prometheus 端点，并用中间件采集请求级指标。
# 指标落 SQLite（解决原实现 model_counts「重启丢失」的问题）。
#
# 采集失败不应影响路由服务：所有监控相关调用都包在 try/except 中。
# ---------------------------------------------------------------------------
_METRICS_ENABLED = os.environ.get("ROUTELLM_METRICS_ENABLED", "1") not in ("0", "false", "False")

if _METRICS_ENABLED:
    try:
        from routellm.monitoring.dashboard import router as _dashboard_router
        from routellm.monitoring.dashboard import set_store as _set_metrics_store
        from routellm.monitoring.middleware import MetricsMiddleware as _MetricsMiddleware
        from routellm.monitoring.store import MetricsStore as _MetricsStore

        _METRICS_STORE = _MetricsStore(
            os.environ.get("ROUTELLM_METRICS_DB", "/tmp/routellm_metrics.db")
        )
        _set_metrics_store(_METRICS_STORE)

        # 网关端口也挂面板路由（便利性；外部访问请用独立的 dashboard 容器）
        app.include_router(_dashboard_router)

        # 配置编辑 API（方案文档 4.8）—— Dashboard 通过它转发编辑请求
        try:
            if _CONFIG_STORE is None:
                raise RuntimeError("运行时配置存储未初始化")
            from routellm.config_runtime import api as _config_api

            _config_api.set_store(_CONFIG_STORE)
            app.include_router(_config_api.router)
            logging.info("配置热更新 API 已挂载: /api/config")
        except Exception as _ce:  # noqa: BLE001
            logging.warning("配置 API 挂载失败: %s", _ce)

        # 中间件需在 app 创建后添加；ctx_var 用本模块的 _METRICS_CTX
        app.add_middleware(_MetricsMiddleware, store=_METRICS_STORE, ctx_var=_METRICS_CTX)

        logging.info(
            "监控已启用: DB=%s, 面板路由已挂载于网关端口, Prometheus=/metrics",
            os.environ.get("ROUTELLM_METRICS_DB", "/tmp/routellm_metrics.db"),
        )
        # 注：Dashboard 已拆为独立容器（见 dashboard/app.py + docker-compose.yml），
        # 不再在网关进程内起第二个端口 —— 同进程双端口会与 Docker 的端口映射
        # 争抢同一端口（实测踩过）。网关这里只保留 /dashboard 路由做便利访问。
    except Exception as _e:  # noqa: BLE001
        logging.warning("监控初始化失败（服务继续运行）: %s", _e)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


async def stream_response(response) -> AsyncGenerator:
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # The model name field contains the parameters for routing.
    # Model name uses format router-[router name]-[threshold] e.g. router-bert-0.7
    # The router type and threshold is used for routing that specific request.
    logging.info(f"Received request: {request}")

    # 采集指标所需的信息写入 request.state（由 MetricsMiddleware 读取）
    _capture_request_meta(request)

    try:
        res = await CONTROLLER.acompletion(
            **request.model_dump(exclude_none=True),
        )
    except RoutingError as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump(),
            status_code=400,
        )

    _capture_response_meta(request, res)

    if request.stream:
        return StreamingResponse(
            content=stream_response(res), media_type="text/event-stream"
        )
    else:
        return JSONResponse(content=res.model_dump())


def _capture_request_meta(request: ChatCompletionRequest) -> None:
    """从请求解析路由名/阈值，并算 prompt hash（隐私：只存 hash）。

    模型名格式：router-<router_name>-<threshold>，如 router-remote_bert-0.5。
    """
    try:
        from routellm.monitoring.metrics import hash_prompt

        parts = str(request.model).split("-")
        threshold = 0.0
        router_name = ""
        if len(parts) >= 3:
            threshold = float(parts[-1])
            router_name = "-".join(parts[1:-1])

        msgs = request.messages
        text = ""
        if isinstance(msgs, str):
            text = msgs
        elif isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                text = str(last.get("content", ""))

        ctx = dict(_METRICS_CTX.get() or {})
        ctx.update(
            router_name=router_name,
            threshold=threshold,
            prompt_hash=hash_prompt(text),
        )
        _METRICS_CTX.set(ctx)
    except Exception:  # noqa: BLE001
        pass  # 指标采集失败不应影响请求


def _capture_response_meta(request: ChatCompletionRequest, res) -> None:
    """提取 token 数、实际路由档位，以及路由元信息（win_rate / 延迟）。"""
    try:
        from routellm.controller import get_routing_info
        from routellm.monitoring.metrics import estimate_cost

        ctx = dict(_METRICS_CTX.get() or {})

        # 路由元信息（由 Controller 在路由决策时记录）
        ri = get_routing_info()
        if ri is not None:
            ctx.update(
                router_name=ri.router,
                threshold=ri.threshold,
                win_rate=ri.win_rate,
                routed_model=ri.routed_model,
                routing_latency_ms=ri.latency_ms,
            )
        else:
            # 兜底：从响应 model 推断档位
            routed = getattr(res, "model", "") or ""
            strong_model = getattr(SETTINGS, "strong_model", "") or ""
            tier = (
                "strong"
                if strong_model and strong_model.split("/")[-1] in routed
                else "weak"
            )
            ctx["routed_model"] = tier

        usage = getattr(res, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        ctx.update(
            prompt_tokens=pt,
            completion_tokens=ct,
            estimated_cost=estimate_cost(ctx.get("routed_model", "weak"), pt, ct),
        )
        _METRICS_CTX.set(ctx)
    except Exception:  # noqa: BLE001
        pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    info: dict = {"status": "online"}
    try:
        from routellm.monitoring.dashboard import get_store

        s = await get_store().summary()
        info["metrics"] = {
            "total_requests": s["total_requests"],
            "strong_ratio": s["strong_ratio"],
            "cost_saved_usd": s["cost_saved_usd"],
            "cache_hit_rate": s["cache_hit_rate"],
        }
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(content=info)


@app.get("/v1/models")
async def list_models():
    """列出可用模型。

    上游缺失此端点（实测 404），而 Cursor / Continue 等客户端在连接时会
    先调 GET /v1/models 预检模型 —— 缺了这个它们连不上。
    见 docs/CHANGELOG.md 问题清单 #3。
    """
    routers = list(CONTROLLER.routers.keys()) if CONTROLLER else list(DEFAULT_ROUTERS)
    data = []
    for name in routers:
        for threshold in DEFAULT_THRESHOLDS:
            data.append(
                {
                    "id": f"router-{name}-{threshold}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "routellm",
                }
            )
    return JSONResponse(content={"object": "list", "data": data})


# 预检时列出的阈值（0.5 为常用默认值）
DEFAULT_THRESHOLDS = [0.5]


def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器。

    注意：不在模块级调用 parse_args() —— 否则任何 `import routellm.openai_server`
    （测试、库引用、gunicorn 加载）都会解析命令行参数并可能 SystemExit。
    这是上游的设计缺陷，见 docs/CHANGELOG.md 问题清单 #7。
    """
    parser = argparse.ArgumentParser(
        description="An OpenAI-compatible API server for LLM routing."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=None,
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument("--base-url", type=str, default=None, help="下游 LLM api_base")
    parser.add_argument("--api-key", type=str, default=None, help="下游 LLM api_key")
    parser.add_argument("--strong-model", type=str, default=None)
    parser.add_argument("--weak-model", type=str, default=None)
    return parser


def _apply_cli_to_env(args: argparse.Namespace) -> None:
    """CLI 参数回填到环境变量，使环境变量成为唯一配置源。

    优先级：环境变量 > 命令行参数 > 默认值
    （显式设置的环境变量不被 CLI 覆盖，便于 Docker 场景以 env 为准）
    """
    mapping = {
        "strong_model": "STRONG_MODEL",
        "weak_model": "WEAK_MODEL",
        "base_url": "API_BASE",
        "api_key": "API_KEY",
        "port": "PORT",
        "host": "HOST",
        "config": "CONFIG",
    }
    for attr, env_name in mapping.items():
        val = getattr(args, attr, None)
        if val is not None and not os.environ.get(f"ROUTELLM_{env_name}"):
            os.environ[f"ROUTELLM_{env_name}"] = str(val)

    if args.routers and not os.environ.get("ROUTELLM_ROUTERS"):
        os.environ["ROUTELLM_ROUTERS"] = ",".join(args.routers)

    if args.verbose:
        os.environ["ROUTELLM_VERBOSE"] = "1"
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    _apply_cli_to_env(_args)

    _boot = Settings.from_env()
    print("Launching server with routers:", _boot.routers)
    # 关键：传 **app 对象** 而非 "routellm.openai_server:app" 字符串。
    # 字符串形式会让 uvicorn 重新 import 本模块，产生两份模块实例 ——
    # lifespan 里的 `global CONTROLLER` 更新的是 __main__ 那份，而
    # 请求处理函数引用的是 routellm.openai_server 那份，导致运行时
    # CONTROLLER 仍为 None（AttributeError: 'NoneType' object has no
    # attribute 'acompletion'）。实测踩过。
    #
    # workers 默认 0 会让 uvicorn 走多进程分支，与上述问题叠加；
    # 这里显式传 None 表示单进程。
    uvicorn.run(
        app,
        port=_boot.port,
        host=_boot.host,
        workers=None,
    )
