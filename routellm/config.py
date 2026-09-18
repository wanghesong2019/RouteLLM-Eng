"""统一配置读取与启动时校验。

对应方案文档"问题5：无部署基建"，以及 docs/CHANGELOG.md 问题清单：

  #1 上游默认模型失效（anyscale provider 已下线）→ 配置外置，可一行替换
  #6 下游模型名需 provider 前缀，否则 litellm 报错

设计原则：
  - 全部配置来自环境变量，不硬编码默认模型名
  - **启动时校验，fail fast**（而非运行时 500）
  - 错误信息明确指出缺什么、怎么修
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# 环境变量前缀
ENV_PREFIX = "ROUTELLM_"

# litellm 支持的 provider 前缀（用于校验下游模型名）
# 完整列表见 https://docs.litellm.ai/docs/providers
KNOWN_PROVIDER_PREFIXES = {
    "openai",
    "azure",
    "anthropic",
    "bedrock",
    "vertex_ai",
    "gemini",
    "cohere",
    "huggingface",
    "together_ai",
    "deepseek",
    "groq",
    "mistral",
    "ollama",
    "vllm",
    "openrouter",
    "xai",
}

DEFAULT_PORT = 6060
DEFAULT_ROUTERS = ["random"]


class ConfigError(Exception):
    """配置缺失或非法。启动时抛出，避免运行到一半才失败。"""


@dataclass
class Settings:
    """服务配置。

    从环境变量读取；缺失必填项时 validate() 抛 ConfigError。
    """

    strong_model: Optional[str] = None
    weak_model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    routers: List[str] = field(default_factory=lambda: list(DEFAULT_ROUTERS))
    port: int = DEFAULT_PORT
    host: str = "0.0.0.0"
    config_path: Optional[str] = None
    inference_url: Optional[str] = None
    # sw_ranking 服务的独立地址（端口 6071）。未设置时回落 inference_url。
    # 之所以需要独立项：remote_bert(6070) 与 remote_sw_ranking(6071) 是
    # 两个独立的 host 服务。
    sw_ranking_inference_url: Optional[str] = None
    verbose: bool = False

    # ---- 容错与 Resilience（方案文档 4.3）----
    # 默认关闭：不改变既有部署行为。开启后下游调用被 重试+熔断+降级 包裹。
    resilience_enabled: bool = False
    # 连续失败多少次开路
    resilience_failure_threshold: int = 5
    # 开路后多久转半开（秒）
    resilience_recovery_timeout: float = 60.0
    # 单次调用的总尝试次数（含首次）
    resilience_max_attempts: int = 3

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "Settings":
        """从环境变量构造。不在此处校验（便于测试单独构造）。"""
        e = env if env is not None else os.environ

        def get(name: str) -> Optional[str]:
            v = e.get(f"{ENV_PREFIX}{name}")
            return v if v not in (None, "") else None

        routers_raw = get("ROUTERS")
        routers = (
            [r.strip() for r in routers_raw.split(",") if r.strip()]
            if routers_raw
            else list(DEFAULT_ROUTERS)
        )

        port_raw = get("PORT")
        try:
            port = int(port_raw) if port_raw else DEFAULT_PORT
        except ValueError as exc:
            raise ConfigError(
                f"{ENV_PREFIX}PORT 必须是整数，当前值: {port_raw!r}"
            ) from exc

        def get_int(name: str, default: int) -> int:
            raw = get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ConfigError(
                    f"{ENV_PREFIX}{name} 必须是整数，当前值: {raw!r}"
                ) from exc

        def get_float(name: str, default: float) -> float:
            raw = get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ConfigError(
                    f"{ENV_PREFIX}{name} 必须是数字，当前值: {raw!r}"
                ) from exc

        def get_bool(name: str, default: bool = False) -> bool:
            raw = get(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            strong_model=get("STRONG_MODEL"),
            weak_model=get("WEAK_MODEL"),
            api_base=get("API_BASE"),
            api_key=get("API_KEY"),
            routers=routers,
            port=port,
            host=get("HOST") or "0.0.0.0",
            config_path=get("CONFIG"),
            inference_url=get("INFERENCE_URL"),
            sw_ranking_inference_url=get("SW_RANKING_INFERENCE_URL"),
            verbose=bool(get("VERBOSE")),
            resilience_enabled=get_bool("RESILIENCE_ENABLED", False),
            resilience_failure_threshold=get_int(
                "RESILIENCE_FAILURE_THRESHOLD", 5
            ),
            resilience_recovery_timeout=get_float(
                "RESILIENCE_RECOVERY_TIMEOUT", 60.0
            ),
            resilience_max_attempts=get_int("RESILIENCE_MAX_ATTEMPTS", 3),
        )

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """校验配置。失败时抛 ConfigError，错误信息指明如何修复。"""
        required = {
            "STRONG_MODEL": self.strong_model,
            "WEAK_MODEL": self.weak_model,
            "API_BASE": self.api_base,
            "API_KEY": self.api_key,
        }
        missing = [f"{ENV_PREFIX}{k}" for k, v in required.items() if not v]
        if missing:
            raise ConfigError(
                "缺少必填配置: "
                + ", ".join(missing)
                + "。请通过环境变量提供，例如："
                + f"{ENV_PREFIX}STRONG_MODEL=openai/qwen3.7-max"
            )

        # 下游模型名必须带 provider 前缀，否则 litellm 抛 BadRequestError
        for label, model in (("STRONG_MODEL", self.strong_model),
                             ("WEAK_MODEL", self.weak_model)):
            self._check_provider_prefix(label, model)

        # 路由器名必须是已注册的
        from routellm.routers.routers import ROUTER_CLS

        unknown = [r for r in self.routers if r not in ROUTER_CLS]
        if unknown:
            raise ConfigError(
                f"未知路由器: {unknown}。可用: {sorted(ROUTER_CLS.keys())}"
            )

        if self.port <= 0 or self.port > 65535:
            raise ConfigError(f"端口非法: {self.port}")

        # 容错参数（方案文档 4.3）：仅在启用时校验，未启用不影响既有行为
        if self.resilience_enabled:
            if self.resilience_failure_threshold <= 0:
                raise ConfigError(
                    f"{ENV_PREFIX}RESILIENCE_FAILURE_THRESHOLD 必须 > 0，"
                    f"当前: {self.resilience_failure_threshold}"
                )
            if self.resilience_recovery_timeout <= 0:
                raise ConfigError(
                    f"{ENV_PREFIX}RESILIENCE_RECOVERY_TIMEOUT 必须 > 0，"
                    f"当前: {self.resilience_recovery_timeout}"
                )
            if self.resilience_max_attempts < 1:
                raise ConfigError(
                    f"{ENV_PREFIX}RESILIENCE_MAX_ATTEMPTS 必须 >= 1，"
                    f"当前: {self.resilience_max_attempts}"
                )

    @staticmethod
    def _check_provider_prefix(label: str, model: Optional[str]) -> None:
        if not model:
            return
        if "/" not in model:
            raise ConfigError(
                f"{ENV_PREFIX}{label}={model!r} 缺少 provider 前缀。"
                f"litellm 要求形如 'openai/<model>' 的写法，"
                f"否则会抛 BadRequestError。"
                f"支持的 provider: {sorted(KNOWN_PROVIDER_PREFIXES)}"
            )
        prefix = model.split("/", 1)[0]
        if prefix not in KNOWN_PROVIDER_PREFIXES:
            raise ConfigError(
                f"{ENV_PREFIX}{label}={model!r} 的 provider 前缀 {prefix!r} 未在已知列表中。"
                f"若确实需要，请将其加入 KNOWN_PROVIDER_PREFIXES。"
                f"已知: {sorted(KNOWN_PROVIDER_PREFIXES)}"
            )

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """用于日志输出，api_key 打码。"""
        def mask(v: Optional[str]) -> Optional[str]:
            return f"{v[:6]}...{v[-4:]}" if v and len(v) > 12 else ("***" if v else None)

        return {
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
            "api_base": self.api_base,
            "api_key": mask(self.api_key),
            "routers": self.routers,
            "host": self.host,
            "port": self.port,
            # 容错状态（方案文档 4.3）——便于运维从启动日志确认是否生效
            "resilience_enabled": self.resilience_enabled,
            "inference_url": self.inference_url,
            "sw_ranking_inference_url": self.sw_ranking_inference_url,
            "config_path": self.config_path,
        }
