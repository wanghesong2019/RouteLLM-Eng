"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
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


app = fastapi.FastAPI(lifespan=lifespan)


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
    try:
        res = await CONTROLLER.acompletion(
            **request.model_dump(exclude_none=True),
        )
    except RoutingError as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump(),
            status_code=400,
        )

    logging.info(CONTROLLER.model_counts)

    if request.stream:
        return StreamingResponse(
            content=stream_response(res), media_type="text/event-stream"
        )
    else:
        return JSONResponse(content=res.model_dump())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})


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
    uvicorn.run(
        "routellm.openai_server:app",
        port=_boot.port,
        host=_boot.host,
        workers=_args.workers,
    )
