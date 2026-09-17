"""网关 API key 鉴权的单元测试（TDD RED 阶段）。

需求
----
网关要绑 0.0.0.0 对外暴露（供 Hermes Agent 等客户端调用），因此需要
API key 鉴权 —— 否则任何人都能白嫖下游 LLM 额度。

用法须与「普通模型 API 服务」一致：
    Authorization: Bearer <key>
即 OpenAI 兼容客户端的标准方式，客户端只需配 api_key 即可，无需改动。

设计
----
- 中间件形式，校验 /v1/* 业务接口
- Key 来源：环境变量 ROUTELLM_GATEWAY_API_KEY（逗号分隔支持多 key）
- **未配置 key 时不校验**（保持向后兼容；内网/回环部署可不设）
- 白名单：/health、/dashboard、/metrics、/api/* 等运维端点不校验
  （面板与健康检查是运维用途，不应要求业务 key）
- 校验失败返回 401，格式与 OpenAI 一致：
      {"error": {"message": "...", "type": "invalid_request_error", "code": "invalid_api_key"}}

运行：
    pytest tests/test_gateway_auth.py -v
"""

import asyncio

import pytest


@pytest.fixture
def auth_mod():
    from routellm.monitoring import auth

    auth.reset()
    return auth


# ------------------------------------------------------------ key 解析


def test_no_key_configured_disables_auth(auth_mod, monkeypatch):
    """未配置 key 时不校验（向后兼容）。"""
    monkeypatch.delenv("ROUTELLM_GATEWAY_API_KEY", raising=False)
    assert auth_mod.get_valid_keys() == []
    assert auth_mod.is_auth_enabled() is False
    assert auth_mod.check_key(None) is True, "未启用鉴权时应一律放行"


def test_single_key(auth_mod, monkeypatch):
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test-123")
    assert auth_mod.get_valid_keys() == ["sk-test-123"]
    assert auth_mod.is_auth_enabled() is True
    assert auth_mod.check_key("sk-test-123") is True
    assert auth_mod.check_key("wrong") is False
    assert auth_mod.check_key(None) is False


def test_multiple_keys_comma_separated(auth_mod, monkeypatch):
    """支持多 key（便于轮换 / 多客户端）。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-a, sk-b ,sk-c")
    keys = auth_mod.get_valid_keys()
    assert keys == ["sk-a", "sk-b", "sk-c"], f"应去除空白: {keys}"
    for k in ("sk-a", "sk-b", "sk-c"):
        assert auth_mod.check_key(k) is True


def test_empty_string_treated_as_disabled(auth_mod, monkeypatch):
    """空字符串视为未配置。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "   ")
    assert auth_mod.get_valid_keys() == []
    assert auth_mod.is_auth_enabled() is False


# ------------------------------------------------------------ header 解析


def test_extract_bearer_token(auth_mod):
    """标准 Bearer 格式（OpenAI 兼容）。"""
    assert auth_mod.extract_token("Bearer sk-abc123") == "sk-abc123"
    assert auth_mod.extract_token("bearer sk-abc123") == "sk-abc123", "大小写不敏感"
    assert auth_mod.extract_token("  Bearer   sk-x  ") == "sk-x", "应容忍空白"


def test_extract_token_invalid_formats(auth_mod):
    assert auth_mod.extract_token(None) is None
    assert auth_mod.extract_token("") is None
    assert auth_mod.extract_token("sk-abc123") is None, "缺少 Bearer 前缀"
    assert auth_mod.extract_token("Basic abc") is None


# ------------------------------------------------------------ 路径白名单


def test_path_whitelist(auth_mod, monkeypatch):
    """运维端点不校验；业务接口校验。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")

    # 业务接口 → 需鉴权
    for p in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
        assert auth_mod.requires_auth(p) is True, f"{p} 应需鉴权"

    # 运维端点 → 免鉴权
    for p in ("/health", "/dashboard", "/metrics", "/api/metrics/summary",
              "/docs", "/openapi.json", "/"):
        assert auth_mod.requires_auth(p) is False, f"{p} 应免鉴权"


def test_whitelist_disabled_when_no_key(auth_mod, monkeypatch):
    """未配 key 时任何路径都不需鉴权。"""
    monkeypatch.delenv("ROUTELLM_GATEWAY_API_KEY", raising=False)
    assert auth_mod.requires_auth("/v1/chat/completions") is False


# ------------------------------------------------------------ 401 响应


def test_unauthorized_response_shape(auth_mod):
    """401 响应体须与 OpenAI 错误格式一致（客户端能正确解析）。"""
    resp = auth_mod.unauthorized_response()
    body = resp.body.decode()
    import json

    d = json.loads(body)
    assert resp.status_code == 401
    err = d.get("error", {})
    assert "message" in err
    assert err.get("code") == "invalid_api_key"
    assert err.get("type") == "invalid_request_error"


# ------------------------------------------------------------ 中间件行为


def _run_middleware(mw, path, headers):
    """用最小 ASGI 环境跑一次中间件，返回 (status, body)。"""
    import json

    captured = {"status": None, "body": b""}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            captured["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            captured["body"] += msg.get("body", b"")

    scope = {
        "type": "http",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "method": "POST",
    }
    asyncio.run(mw(scope, receive, send))
    return captured["status"], captured["body"]


def test_middleware_blocks_without_key(auth_mod, monkeypatch):
    """启用鉴权后，无 key 的业务请求应被 401 拦下。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")
    mw = auth_mod.ApiKeyMiddleware(app=None)
    status, body = _run_middleware(mw, "/v1/chat/completions", {})
    assert status == 401


def test_middleware_allows_with_correct_key(auth_mod, monkeypatch):
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")
    mw = auth_mod.ApiKeyMiddleware(app=None)
    status, _ = _run_middleware(
        mw, "/v1/chat/completions", {"authorization": "Bearer sk-test"}
    )
    assert status == 200


def test_middleware_passes_whitelisted_path(auth_mod, monkeypatch):
    """/health 等运维端点即使无 key 也应放行。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")
    mw = auth_mod.ApiKeyMiddleware(app=None)
    status, _ = _run_middleware(mw, "/health", {})
    assert status == 200, "/health 应免鉴权（容器探针用）"


def test_middleware_disabled_without_key_config(auth_mod, monkeypatch):
    """未配 key 时中间件不拦截（向后兼容）。"""
    monkeypatch.delenv("ROUTELLM_GATEWAY_API_KEY", raising=False)
    mw = auth_mod.ApiKeyMiddleware(app=None)
    status, _ = _run_middleware(mw, "/v1/chat/completions", {})
    assert status == 200


def test_lifespan_scope_not_blocked(auth_mod, monkeypatch):
    """lifespan scope 必须透传，不能被鉴权拦截。

    这是个致命的不变量：若鉴权中间件拦截 lifespan，应用的启动逻辑
    （CONTROLLER 初始化等）永不执行，业务接口全部 500。
    日志表现为 "ASGI 'lifespan' protocol appears unsupported"。

    lifespan scope 的特征：type="lifespan"，**没有 path 字段**。
    若不显式排除 type != "http"，requires_auth("") 会返回 True。
    """
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")

    passed = {"hit": False, "scope_type": None}

    async def downstream(scope, receive, send):
        passed["hit"] = True
        passed["scope_type"] = scope.get("type")

    mw = auth_mod.ApiKeyMiddleware(app=downstream)

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(msg):
        pass

    scope = {"type": "lifespan"}  # 注意：无 path
    asyncio.run(mw(scope, receive, send))

    assert passed["hit"] is True, "lifespan scope 被拦截了 —— 启动逻辑将永不执行"
    assert passed["scope_type"] == "lifespan"


def test_websocket_scope_not_blocked(auth_mod, monkeypatch):
    """非 HTTP 协议（如 websocket）也应透传，不做鉴权判断。"""
    monkeypatch.setenv("ROUTELLM_GATEWAY_API_KEY", "sk-test")

    passed = {"hit": False}

    async def downstream(scope, receive, send):
        passed["hit"] = True

    mw = auth_mod.ApiKeyMiddleware(app=downstream)

    async def receive():
        return {}

    async def send(msg):
        pass

    asyncio.run(mw({"type": "websocket", "path": "/ws"}, receive, send))
    assert passed["hit"] is True
