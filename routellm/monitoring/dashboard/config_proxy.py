"""Dashboard 配置转发层。

方案 B（用户选定）：**配置存网关，Dashboard 通过网关 API 转发**。

    浏览器 → Dashboard(:8092) /api/config → 网关(:6060) /api/config

为何用转发而非让 Dashboard 直接持有配置
--------------------------------------
1. **配置归网关所有**（single source of truth）—— Dashboard 只是 UI
2. **保持面板的只读边界** —— metrics 目录以 `:ro` 挂载；不给它配置写权限
3. **转发层可集中做错误归一化**：网关不可达 / key 错误 / 5xx 都转成
   前端可展示的结构化错误，避免面板因后端异常而整页崩掉

配置（环境变量）
----------------
    ROUTELLM_GATEWAY_URL       网关地址，默认 http://host.docker.internal:6060
    ROUTELLM_GATEWAY_API_KEY   调用网关的 Bearer key（与客户端用的是同一个）

安全：**前端不需要知道网关 key** —— key 只存在于 Dashboard 容器的环境变量里，
浏览器只与 Dashboard 通信。这比让浏览器直连网关（方案C）更安全。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_GATEWAY_URL = "http://host.docker.internal:6060"
DEFAULT_TIMEOUT = 15.0


def reset() -> None:
    """重置（测试用）。当前无缓存状态，保留接口以便将来扩展。"""
    return None


def get_gateway_url() -> str:
    """网关地址（去掉尾部斜杠）。"""
    url = os.environ.get("ROUTELLM_GATEWAY_URL") or DEFAULT_GATEWAY_URL
    return url.rstrip("/")


def auth_header() -> Dict[str, str]:
    """构造调用网关的鉴权头。未配 key 时不带（网关未启用鉴权时可用）。"""
    key = os.environ.get("ROUTELLM_GATEWAY_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


async def _request(
    method: str,
    path: str,
    body: Optional[dict] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[int, dict]:
    """向网关发请求。返回 (status_code, json)。

    异常由调用方捕获（forward_* 会转成结构化错误）。
    """
    import httpx

    url = get_gateway_url() + path
    headers = {"Content-Type": "application/json", **auth_header()}

    async with httpx.AsyncClient(timeout=timeout) as client:
        if method == "GET":
            r = await client.get(url, headers=headers)
        elif method == "PUT":
            r = await client.put(url, headers=headers, json=body or {})
        elif method == "POST":
            r = await client.post(url, headers=headers, json=body or {})
        else:
            raise ValueError(f"不支持的方法: {method}")

    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        data = {"detail": r.text[:500]}
    return r.status_code, data


def _raise_for_client_error(status: int, data: Any) -> None:
    """4xx 客户端错误：抛 HTTPException **保留状态码**。

    为何不能像 5xx 那样归一化成 200 + proxy_error
    --------------------------------------------
    归一化的初衷是"面板不因网关异常整页崩"，对 5xx（网关自身故障）成立；
    但 4xx 表示**请求本身有问题**（参数非法 / 鉴权失败 / 资源不存在），
    此时把状态码抹成 200 会让前端无法区分"成功"和"参数错"。

    实测暴露路径：`POST /api/config/verify?tier=medium`
        网关 → 422 {"detail": "未知 tier"}
        面板（旧）→ HTTP 200 {"proxy_error": ...}  ← 前端误判为成功

    详情原样透传（含网关的 detail），前端才能给出具体提示。
    """
    from fastapi import HTTPException

    raise HTTPException(status_code=status, detail=_extract_detail(data))


def _extract_detail(data: Any) -> Any:
    """从网关响应里取出可读的错误详情（优先 detail，其次整体）。"""
    if isinstance(data, dict) and "detail" in data:
        return data["detail"]
    if isinstance(data, dict) and "error" in data:
        return data["error"]
    return data


def _error_payload(exc: Optional[Exception] = None, status: Optional[int] = None) -> Dict[str, Any]:
    """把 5xx / 不可达统一成前端可展示的结构（面板不崩）。"""
    if exc is not None:
        return {
            "proxy_error": f"{type(exc).__name__}: {exc}",
            "proxy_hint": "无法连接网关，请检查 ROUTELLM_GATEWAY_URL 与网关状态",
        }
    if status == 401:
        return {
            "proxy_error": "401 Unauthorized",
            "proxy_hint": "网关拒绝了鉴权：请检查 ROUTELLM_GATEWAY_API_KEY 是否与网关一致",
        }
    return {
        "proxy_error": f"网关返回 HTTP {status}",
        "proxy_hint": "请查看网关日志",
    }


def _handle_error_status(status: int, data: Any, where: str) -> Dict[str, Any]:
    """按状态码分派错误处理。

    4xx 客户端错误 → 抛 HTTPException（保留状态码，见 `_raise_for_client_error`）
    5xx 服务端错误 → 返回归一化结构（面板不崩）

    Returns:
        5xx 时的归一化错误结构。4xx 时不返回（抛异常）。
    """
    logger.warning("转发 %s 网关返回 %s: %s", where, status, str(data)[:200])
    if 400 <= status < 500:
        _raise_for_client_error(status, data)
    return {**_error_payload(status=status), "proxy_status": status}


async def forward_get_config() -> Dict[str, Any]:
    """转发 GET /api/config。

    4xx 抛 HTTPException（保留状态码）；5xx / 不可达返回 proxy_error 结构。
    """
    try:
        status, data = await _request("GET", "/api/config")
    except Exception as e:  # noqa: BLE001
        logger.warning("转发配置读取失败: %s", e)
        return _error_payload(exc=e)

    if status >= 400:
        return _handle_error_status(status, data, "GET /api/config")

    if isinstance(data, dict):
        data.setdefault("proxy_error", None)
    return data


async def forward_put_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """转发 PUT /api/config。

    4xx 抛 HTTPException（如未知字段 422）；5xx / 不可达归一化。
    """
    try:
        status, data = await _request("PUT", "/api/config", body=payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("转发配置更新失败: %s", e)
        return _error_payload(exc=e)

    if status >= 400:
        return _handle_error_status(status, data, "PUT /api/config")

    if isinstance(data, dict):
        data.setdefault("proxy_error", None)
    return data


async def forward_verify(tier: Optional[str] = None) -> Dict[str, Any]:
    """转发 POST /api/config/verify（连通性预检）。

    Args:
        tier: "strong" / "weak" —— 按侧测试；None 表示测全局（向后兼容）。
            透传给网关，由网关决定用哪一侧的 base_url / api_key。

    4xx 抛 HTTPException（如非法 tier → 422）；5xx / 不可达归一化。
    """
    path = "/api/config/verify"
    if tier:
        path += f"?tier={tier}"
    try:
        status, data = await _request("POST", path)
    except Exception as e:  # noqa: BLE001
        logger.warning("转发连通性预检失败: %s", e)
        return _error_payload(exc=e)

    if status >= 400:
        return _handle_error_status(status, data, "POST /api/config/verify")
    return data



# --------------------------------------------------------------------------- 路由


@router.get("/api/config")
async def proxy_get_config() -> Dict[str, Any]:
    """读取配置（转发到网关）。"""
    return await forward_get_config()


@router.put("/api/config")
async def proxy_put_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """更新配置（转发到网关，立即生效）。"""
    return await forward_put_config(payload)


@router.post("/api/config/verify")
async def proxy_verify(tier: Optional[str] = None) -> Dict[str, Any]:
    """连通性预检（转发到网关）。

    tier 查询参数按侧透传 —— 强弱可能接不同 provider，各测各的。
    """
    return await forward_verify(tier=tier)



@router.get("/api/config/gateway")
async def proxy_gateway_info() -> Dict[str, Any]:
    """暴露转发目标信息（供前端显示"当前编辑的是哪个网关"）。"""
    url = get_gateway_url()
    has_key = bool(auth_header())
    return {
        "gateway_url": url,
        "auth_configured": has_key,
        "reachable": None,  # 前端可调 /api/config 实际验证
    }
