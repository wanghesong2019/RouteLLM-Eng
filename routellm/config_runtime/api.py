"""运行时配置编辑 API（网关侧）。

方案文档 4.8 节 —— 方案B：**配置存网关，Dashboard 通过本 API 转发**。

    网关持有配置真相（single source of truth）；
    Dashboard 只是 UI 层，不持有配置。

接口
----
    GET  /api/config          掩码视图（不回显密钥明文）
    PUT  /api/config          更新配置（立即生效）
    POST /api/config/verify   连通性预检

安全设计
--------
1. **鉴权**：路径 `/api/config` 不在 ApiKeyMiddleware 的白名单里吗？
   —— 需要确认。白名单当前含 `/api/` 前缀（为运维面板放行）。
   配置编辑**必须鉴权**（改 key = 改成本中心），故本模块自带一个
   校验开关：`ROUTELLM_CONFIG_REQUIRE_AUTH`（默认开），
   在 API 层做二次校验，不依赖中间件路径规则。
2. **不回显明文**：GET 走 `store.masked()`；PUT 响应同样掩码。
3. **审计日志**：记录变更字段与时间，**不记录密钥明文**。
4. **连通性预检**：可选。改 base_url 前试探下游，避免改坏服务。

为何要做连通性预检
------------------
改坏 base_url 会导致**所有**下游调用失败 —— 影响面是整个网关。
预检用一次轻量请求（models 列表）验证 (base_url, api_key, model) 组合可用，
失败则拒绝写入。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from routellm.config_runtime.store import RuntimeConfigStore, mask_secret

logger = logging.getLogger(__name__)

router = APIRouter()

_STORE: Optional[RuntimeConfigStore] = None

# 可编辑字段（前端据此渲染表单；其余配置项为只读，改动需重启）
EDITABLE_FIELDS = ("strong_model", "weak_model", "api_base", "api_key")


def set_store(store: RuntimeConfigStore) -> None:
    global _STORE
    _STORE = store


def get_store() -> RuntimeConfigStore:
    global _STORE
    if _STORE is None:
        _STORE = RuntimeConfigStore()
    return _STORE


# --------------------------------------------------------------------------- 模型


class ConfigUpdate(BaseModel):
    """配置更新请求。

    所有字段可选 —— 未传或 None 表示"不修改"（避免前端未填字段清空配置）。

    extra="forbid"：未知字段直接报错，防止拼写错误静默失效。
    """

    model_config = ConfigDict(extra="forbid")

    strong_model: Optional[str] = Field(None, description="强模型名（含 provider 前缀）")
    weak_model: Optional[str] = Field(None, description="弱模型名（含 provider 前缀）")
    api_base: Optional[str] = Field(None, description="下游 LLM API 地址")
    api_key: Optional[str] = Field(None, description="下游 LLM API key")
    verify: bool = Field(
        False,
        description="是否先做连通性预检；失败则拒绝写入（避免改坏服务）",
    )


# --------------------------------------------------------------------------- 鉴权


def _requires_auth() -> bool:
    """配置编辑是否要求鉴权。

    默认开启。设为 ROUTELLM_CONFIG_REQUIRE_AUTH=0 可关闭（仅限内网调试）。
    """
    return os.environ.get("ROUTELLM_CONFIG_REQUIRE_AUTH", "1") not in ("0", "false", "False")


def _check_auth(request: Optional[Request]) -> None:
    """在 API 层做二次鉴权校验。

    不依赖中间件的路径白名单规则（`/api/` 前缀当前对运维面板放行），
    因此这里显式校验，确保配置编辑始终受保护。
    """
    if not _requires_auth():
        return
    if request is None:
        return  # 单元测试直调（无 Request 对象）

    from routellm.monitoring.auth import check_key, extract_token, is_auth_enabled

    if not is_auth_enabled():
        # 未配置网关 key → 无从校验；此时放行但告警（与整体鉴权策略一致）
        logger.warning("配置编辑未鉴权：未配置 ROUTELLM_GATEWAY_API_KEY")
        return

    token = extract_token(request.headers.get("authorization"))
    if not check_key(token):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "配置编辑需要有效的 API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


# --------------------------------------------------------------------------- 连通性预检


async def _probe_downstream(
    base_url: str, api_key: str, model: str, timeout: float = 5.0
) -> Tuple[bool, str]:
    """试探下游是否可用（轻量请求）。

    用 /models 端点（比 chat/completions 便宜，且能同时验证 base_url 与 key）。
    返回 (是否成功, 详情)。
    """
    import httpx

    if not base_url:
        return False, "api_base 为空"

    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code < 400:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- 接口


@router.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """读取当前配置（密钥掩码）。"""
    store = get_store()
    res = store.masked()
    res["editable_fields"] = list(EDITABLE_FIELDS)
    res["note"] = "api_key 为掩码显示；提交时留空表示不修改"
    return res


@router.put("/api/config")
async def put_config(update: ConfigUpdate, request: Request = None) -> Dict[str, Any]:
    """更新配置并立即生效（无需重启）。

    若 `verify=True` 且提供了 api_base/api_key，会先做连通性预检；
    预检失败则拒绝写入（避免改坏服务导致全部下游调用失败）。
    """
    _check_auth(request)
    store = get_store()

    # 只取可编辑字段（排除 verify 等控制位，它们不是配置项）
    fields = {
        k: v
        for k, v in update.model_dump(exclude_none=True).items()
        if k in EDITABLE_FIELDS
    }

    if not fields:
        return {"ok": True, "message": "无变更", **store.masked()}

    # 连通性预检（可选）
    verify = getattr(update, "verify", False)
    if verify:
        current = store.load()
        base = fields.get("api_base", current.api_base)
        key = fields.get("api_key", current.api_key)
        model = fields.get("strong_model", current.strong_model)
        ok, detail = await _probe_downstream(base, key, model)
        if not ok:
            logger.warning("配置预检失败，拒绝写入: %s", detail)
            return {
                "ok": False,
                "message": f"连通性预检失败，配置未生效: {detail}",
                **store.masked(),
            }

    old = store.load()
    new = store.update(**fields)

    # 审计日志（只记字段名，不记值 —— 值可能含密钥）
    changed = [k for k in fields if getattr(old, k, None) != getattr(new, k, None)]
    if changed:
        logger.info(
            "配置变更生效: fields=%s at=%s", ",".join(changed), time.strftime("%Y-%m-%d %H:%M:%S")
        )

    res = store.masked()
    res["ok"] = True
    res["changed"] = changed
    return res


@router.post("/api/config/verify")
async def verify_connectivity(request: Request = None) -> Dict[str, Any]:
    """连通性预检（不修改配置）。

    供前端"测试连接"按钮使用，也供 PUT 前的可选校验。
    """
    _check_auth(request)
    store = get_store()
    cfg = store.load()
    ok, detail = await _probe_downstream(cfg.api_base, cfg.api_key, cfg.strong_model)
    return {
        "ok": ok,
        "detail": detail,
        "probed": {
            "api_base": cfg.api_base,
            "strong_model": cfg.strong_model,
            "api_key": mask_secret(cfg.api_key),
        },
    }
