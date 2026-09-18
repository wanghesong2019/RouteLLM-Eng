"""tools 字段类型校验测试（TDD RED→GREEN）。

RouteLLM 网关的 ChatCompletionRequest.tools 类型定义过窄
（Dict[str, Union[str, int, float]]），不支持标准 OpenAI tools 格式中
function 字段的嵌套对象，导致 422。

运行：
    pytest tests/test_tools_field_validation.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """网关 TestClient（关闭监控，避免后台线程/DB 干扰）。"""
    monkeypatch.setenv("ROUTELLM_METRICS_ENABLED", "0")
    monkeypatch.setenv("ROUTELLM_DASHBOARD_ENABLED", "0")
    monkeypatch.setenv("ROUTELLM_API_KEY", "sk-dum...ream")

    import importlib

    import routellm.openai_server as srv

    importlib.reload(srv)
    return TestClient(srv.app, raise_server_exceptions=False)


def test_tools_with_nested_function_object_not_422(client):
    """标准 OpenAI tools 格式（function 为嵌套对象）不应触发 422。

    Hermes 发的 tools 格式：
        [{"type": "function",
          "function": {"name": "clarify", "description": "...",
                        "parameters": {"type": "object", ...}}}]
    """
    standard_tools = [
        {
            "type": "function",
            "function": {
                "name": "clarify",
                "description": "Ask the user a question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                },
            },
        }
    ]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "router-remote_bert-0.5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": standard_tools,
        },
    )
    # 不应 422 —— 鉴权通过后可能因下游不可达而 500/其他，但绝不是 422
    assert r.status_code != 422, (
        f"标准 OpenAI tools 格式不应被 422 拒绝: {r.status_code}: {r.text[:500]}"
    )
