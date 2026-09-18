"""网关 API key 鉴权。

背景
----
网关要绑 0.0.0.0 对外暴露（供 Hermes Agent 等客户端调用），必须鉴权 ——
否则任何人都能白嫖下游 LLM 额度（真金白银）。

用法（与普通模型 API 服务一致）
------------------------------
客户端只需像配 OpenAI 一样填 api_key，走标准 header：

    Authorization: Bearer <ROUTELLM_GATEWAY_API_KEY>

无需任何定制客户端。Hermes Agent / openai SDK / curl 都能直接用。

配置
----
    ROUTELLM_GATEWAY_API_KEY=sk-xxx             单 key
    ROUTELLM_GATEWAY_API_KEY=sk-a,sk-b          多 key（轮换/多客户端）
    （未设置或为空）                              **不鉴权**（向后兼容）

安全约定
--------
- 白名单端点免鉴权：/health（容器探针）、/dashboard、/config、/metrics、
  /api/*（运维面板）
  —— 这些是运维用途，不应要求业务 key，否则探针和面板都不可用。
  注意 /dashboard 与 /config 都是面板页面路由（导航链接整页跳转带不了
  Authorization），必须成对列入，否则面板导航点击即 401。
- 业务接口（/v1/*）一律校验。
- 401 响应体沿用 OpenAI 错误格式，客户端能正确解析。
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

ENV_KEY = "ROUTELLM_GATEWAY_API_KEY"

# 免鉴权路径（精确匹配）与前缀（前缀匹配）
#
# /config 与 /dashboard 必须成对出现：这两个都是面板自身的页面路由
# （dashboard 页面导航里有 <a href="/config">运行时配置</a>）。
# 浏览器整页跳转带不了 Authorization header，所以面板页面一旦不在白名单，
# 用户点击导航就会看到 401 invalid_api_key —— 实测公网部署时踩到过。
# 注：真正读写配置的 /api/config 由 WHITELIST_PREFIX 的 "/api/" 覆盖。
WHITELIST_EXACT = {"/", "/health", "/dashboard", "/config", "/metrics",
                   "/docs", "/openapi.json", "/redoc"}
WHITELIST_PREFIX = ("/api/", "/docs/", "/static/")


def get_valid_keys() -> List[str]:
    """解析配置的合法 key 列表（逗号分隔，去空白）。空 → 未启用鉴权。"""
    raw = os.environ.get(ENV_KEY, "") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


def is_auth_enabled() -> bool:
    """是否启用鉴权（配置了至少一个 key）。"""
    return len(get_valid_keys()) > 0


def reset() -> None:
    """重置（测试用）。当前无常驻状态，保留接口以便将来扩展。"""
    return None


def requires_auth(path: str) -> bool:
    """该路径是否需要鉴权。

    未启用鉴权时一律 False（向后兼容）。
    """
    if not is_auth_enabled():
        return False
    if path in WHITELIST_EXACT:
        return False
    for prefix in WHITELIST_PREFIX:
        if path.startswith(prefix):
            return False
    return True


def extract_token(auth_header: Optional[str]) -> Optional[str]:
    """从 Authorization header 提取 Bearer token。

    容忍大小写与前导/尾随空白；格式不符返回 None。
    """
    if not auth_header:
        return None
    parts = auth_header.strip().split()
    if len(parts) != 2:
        return None
    scheme, token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def check_key(token: Optional[str]) -> bool:
    """校验 token 是否合法。

    未启用鉴权时一律放行。比较用 hmac.compare_digest 防时序侧信道。
    """
    keys = get_valid_keys()
    if not keys:
        return True
    if token is None:
        return False
    return any(hmac.compare_digest(token, k) for k in keys)


class _UnauthorizedResponse:
    """构造 401 响应（OpenAI 兼容格式）。

    用一个轻量对象而非直接返回 Starlette Response —— 便于单元测试
    检查 status_code / body，也避免在无 ASGI 环境时引入依赖。
    """

    status_code = 401

    def __init__(self, message: str = "Incorrect API key provided."):
        self._payload = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
        self.body = json.dumps(self._payload).encode()
        self.headers = {"content-type": "application/json"}


def unauthorized_response() -> _UnauthorizedResponse:
    """标准 401 响应对象。"""
    return _UnauthorizedResponse()


class ApiKeyMiddleware:
    """校验 /v1/* 请求的 Authorization header。

    Args:
        app: 下游 ASGI 应用。
        enabled: 可选覆盖（默认按环境变量判断）。

    行为：
        - 未配置 key → 直接放行（向后兼容）
        - 白名单路径 → 放行
        - /v1/* 且 key 不合法 → 401（OpenAI 格式）
    """

    def __init__(self, app=None, enabled: Optional[bool] = None):
        self.app = app
        self._enabled_override = enabled

    def _enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return is_auth_enabled()

    async def __call__(self, scope, receive, send):
        # 关键：只拦截 HTTP 请求。必须显式判断 scope 类型 ——
        # lifespan 等非 HTTP 协议的 scope 没有 path，若不排除会被
        # requires_auth("") 判为需鉴权，导致 lifespan 被 401 拦截，
        # 应用的启动逻辑（CONTROLLER 初始化）永不执行
        # （实测踩过：日志表现为 "ASGI 'lifespan' protocol appears
        #  unsupported" + 业务全部 500）。
        if scope.get("type") != "http":
            if self.app is not None:
                return await self.app(scope, receive, send)
            return

        # app 为 None 仅出现在单元测试（直接调中间件验证拦截行为）
        if not self._enabled() or not requires_auth(scope.get("path", "")):
            if self.app is not None:
                return await self.app(scope, receive, send)
            return await self._passthrough(send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        token = extract_token(headers.get("authorization"))

        if check_key(token):
            if self.app is not None:
                return await self.app(scope, receive, send)
            return await self._passthrough(send)

        logger.warning(
            "鉴权失败: path=%s (key %s)",
            scope.get("path", ""),
            "缺失" if token is None else "不匹配",
        )
        resp = unauthorized_response()
        await send({
            "type": "http.response.start",
            "status": resp.status_code,
            "headers": [(k.encode(), v.encode()) for k, v in resp.headers.items()],
        })
        await send({"type": "http.response.body", "body": resp.body})

    @staticmethod
    async def _passthrough(send):
        """无下游 app 时的放行响应（测试用）。"""
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})
