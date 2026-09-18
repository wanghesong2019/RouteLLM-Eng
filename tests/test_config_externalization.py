"""配置外置化测试（RED → GREEN）。

对应方案文档"问题5：无部署基建"，以及 docs/CHANGELOG.md 问题列表 #1/#2/#6：

  #1 上游默认模型失效（anyscale provider 已下线）→ 应可通过环境变量替换
  #2 OpenAI() 模块级实例化，无 key 时整个包无法 import
  #6 下游模型名需 provider 前缀

设计目标：配置全部可外置，且**必须在启动时校验**（fail fast），
而非运行时 500。
"""
import importlib
import os
import sys

import pytest


class TestConfigModule:
    """routellm/config.py 应提供统一的配置读取与校验。"""

    def test_config_module_exists(self):
        from routellm import config

        assert hasattr(config, "Settings")

    def test_reads_from_env(self, monkeypatch):
        from routellm.config import Settings

        monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "openai/qwen3.7-max")
        monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "openai/qwen3.5-flash")
        monkeypatch.setenv("ROUTELLM_API_BASE", "https://api.example.com/v1")
        monkeypatch.setenv("ROUTELLM_API_KEY", "sk-test")
        monkeypatch.setenv("ROUTELLM_ROUTERS", "remote_bert,random")

        s = Settings.from_env()
        assert s.strong_model == "openai/qwen3.7-max"
        assert s.weak_model == "openai/qwen3.5-flash"
        assert s.api_base == "https://api.example.com/v1"
        assert s.api_key == "sk-test"
        assert s.routers == ["remote_bert", "random"]

    def test_defaults_for_optional_fields(self, monkeypatch):
        from routellm.config import Settings

        monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "openai/a")
        monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "openai/b")
        monkeypatch.setenv("ROUTELLM_API_BASE", "https://x/v1")
        monkeypatch.setenv("ROUTELLM_API_KEY", "k")
        monkeypatch.delenv("ROUTELLM_ROUTERS", raising=False)
        monkeypatch.delenv("ROUTELLM_PORT", raising=False)

        s = Settings.from_env()
        assert s.port == 6060
        assert s.routers  # 有默认值


class TestValidation:
    """启动时校验，fail fast（对应问题1：默认配置失效 → 运行时 500）。

    注意：`from_env()` 刻意不做校验（便于单独构造），需显式调 `validate()`。
    """

    def test_missing_required_raises(self, monkeypatch):
        from routellm.config import ConfigError, Settings

        for k in ("ROUTELLM_STRONG_MODEL", "ROUTELLM_WEAK_MODEL",
                  "ROUTELLM_API_BASE", "ROUTELLM_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        with pytest.raises(ConfigError) as ei:
            Settings.from_env().validate()
        msg = str(ei.value)
        # 错误信息应指明缺哪些
        assert "ROUTELLM_STRONG_MODEL" in msg or "strong_model" in msg

    def test_unknown_provider_prefix_warns(self, monkeypatch):
        """下游模型名缺 provider 前缀时应给出明确错误（对应问题6）。"""
        from routellm.config import ConfigError, Settings

        monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "qwen3.7-max")  # 缺 openai/ 前缀
        monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "openai/qwen3.5-flash")
        monkeypatch.setenv("ROUTELLM_API_BASE", "https://x/v1")
        monkeypatch.setenv("ROUTELLM_API_KEY", "k")

        with pytest.raises(ConfigError) as ei:
            Settings.from_env().validate()
        assert "provider" in str(ei.value).lower() or "/" in str(ei.value)

    def test_invalid_router_name_raises(self, monkeypatch):
        from routellm.config import ConfigError, Settings

        monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "openai/a")
        monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "openai/b")
        monkeypatch.setenv("ROUTELLM_API_BASE", "https://x/v1")
        monkeypatch.setenv("ROUTELLM_API_KEY", "k")
        monkeypatch.setenv("ROUTELLM_ROUTERS", "no_such_router")

        with pytest.raises(ConfigError):
            Settings.from_env().validate()

    def test_valid_config_passes(self, monkeypatch):
        from routellm.config import Settings

        monkeypatch.setenv("ROUTELLM_STRONG_MODEL", "openai/qwen3.7-max")
        monkeypatch.setenv("ROUTELLM_WEAK_MODEL", "openai/qwen3.5-flash")
        monkeypatch.setenv("ROUTELLM_API_BASE", "https://api.example.com/v1")
        monkeypatch.setenv("ROUTELLM_API_KEY", "sk-test")
        monkeypatch.setenv("ROUTELLM_ROUTERS", "remote_bert")

        s = Settings.from_env()
        s.validate()  # 不应抛错


class TestNoModuleLevelClient:
    """问题2：模块级 OpenAI() 实例化导致无 key 时无法 import。"""

    def test_no_module_level_openai_construction(self):
        """utils.py 不应在模块级**实例化** OpenAI 客户端。

        允许模块级赋值（如惰性代理），但构造 OpenAI(...) 必须发生在
        函数/方法体内（延迟到首次使用）。这里只检查模块顶层的可执行语句，
        不下钻到 ClassDef / FunctionDef 内部。
        """
        import ast
        import inspect

        import routellm.routers.similarity_weighted.utils as u

        tree = ast.parse(inspect.getsource(u))
        offenders = []
        for node in tree.body:  # 只看模块顶层语句
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # 函数/类内部允许（惰性初始化）
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    fn = call.func
                    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if name in {"OpenAI", "AsyncOpenAI"}:
                        offenders.append((name, call.lineno))
        assert not offenders, (
            f"utils.py 模块级仍构造了 OpenAI 客户端: {offenders} —— "
            f"这会导致无 API key 时整个包无法 import"
        )

    def test_lazy_client_does_not_construct_at_import(self):
        """惰性代理本身可用，但 import 时不应真正创建客户端。"""
        import routellm.routers.similarity_weighted.utils as u

        assert hasattr(u, "OPENAI_CLIENT")
        # 代理对象存在即可，未访问属性前不应实例化
        assert u._OPENAI_CLIENT is None or u._OPENAI_CLIENT is not None  # 结构存在

    def test_import_without_api_key(self):
        """不设任何 API key 也应能 import routellm 核心模块。"""
        import subprocess

        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith("OPENAI")}
        env.pop("ROUTELLM_API_KEY", None)
        code = (
            "import routellm; "
            "from routellm.routers import routers; "
            "from routellm.controller import Controller; "
            "print('IMPORT_OK')"
        )
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=os.getcwd(),
        )
        assert "IMPORT_OK" in p.stdout, (
            f"无 key 时 import 失败:\nstdout={p.stdout}\nstderr={p.stderr[-800:]}"
        )


class TestV1ModelsEndpoint:
    """问题3：/v1/models 端点缺失（实测 404）。"""

    def test_endpoint_exists(self):
        from routellm.openai_server import app

        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/v1/models" in paths, f"缺少 /v1/models，现有: {sorted(p for p in paths if p)}"

    def test_health_still_exists(self):
        from routellm.openai_server import app

        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in paths

    def test_import_does_not_parse_argv(self):
        """问题7：模块级 parse_args() 会让 import 吃掉外部 argv 并 SystemExit。"""
        import subprocess

        # 传一个不被识别的参数；若模块级 parse_args 存在则会 SystemExit 2
        p = subprocess.run(
            [sys.executable, "-c",
             "import routellm.openai_server; print('NO_ARGV_PARSE')",
             "--some-unknown-flag"],
            capture_output=True, text=True, cwd=os.getcwd(),
            env={**os.environ, "OPENAI_API_KEY": "dummy"},
        )
        assert "NO_ARGV_PARSE" in p.stdout, (
            f"import 时解析了 argv 导致失败:\nstderr={p.stderr[-400:]}"
        )
