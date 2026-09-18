"""惰性导入测试：容器内不装 torch/transformers/datasets 也能用 remote_bert。

背景（见 docs/CHANGELOG.md）：
  routers.py 在模块级 import torch / transformers / datasets，
  导致容器即使只用 remote_bert（HTTP 调用 host 推理服务），
  也必须安装 554MB 的 torch —— 镜像膨胀到 3GB+，构建耗时近 1 小时。

解决：把这些 import 改为按需（惰性）导入。
  使用 remote_bert / random 时不触发；使用 bert / causal_llm / sw_ranking / mf 时才触发。
"""
import ast
import inspect
import subprocess
import sys

HEAVY_MODULES = {"torch", "transformers", "datasets", "huggingface_hub"}


class TestLazyImports:
    def test_routers_no_module_level_heavy_imports(self):
        """routers.py 不应在模块级导入重依赖。"""
        import routellm.routers.routers as m

        tree = ast.parse(inspect.getsource(m))
        offenders = []
        for node in tree.body:  # 只看模块顶层
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in HEAVY_MODULES:
                        offenders.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in HEAVY_MODULES:
                    offenders.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
        assert not offenders, (
            f"routers.py 模块级仍导入重依赖: {offenders}"
        )

    def test_remote_only_import_without_heavy_deps(self):
        """模拟容器环境：屏蔽 torch/transformers/datasets 后仍能 import 并使用 remote_bert。

        通过 sys.modules 注入哨兵，使 import 这些包直接失败。
        """
        code = """
import sys
class _Blocker:
    def find_module(self, name, path=None):
        root = name.split('.')[0]
        if root in {'torch', 'transformers', 'datasets', 'huggingface_hub'}:
            return self
        return None
    def load_module(self, name):
        raise ImportError(f'BLOCKED_HEAVY_DEP: {name}')
sys.meta_path.insert(0, _Blocker())

from routellm.routers.routers import ROUTER_CLS
assert 'remote_bert' in ROUTER_CLS, 'remote_bert 未注册'
assert 'random' in ROUTER_CLS, 'random 未注册'

from routellm.controller import Controller
c = Controller(
    routers=['remote_bert'],
    strong_model='openai/strong',
    weak_model='openai/weak',
    config={'remote_bert': {'base_url': 'http://127.0.0.1:6070'}},
)

# 容错模块（方案文档 4.3）不得引入重型依赖：
# resilience/ 按异常类名语义判定可重试性，刻意不 import litellm。
import routellm.resilience as rz
assert rz.CircuitBreaker and rz.ResilientCaller, 'resilience 未导出核心类'
_ = rz.CircuitBreaker(failure_threshold=2, recovery_timeout=1)
print('LAZY_IMPORT_OK')
"""
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=__import__("os").getcwd(),
            env={**__import__("os").environ, "OPENAI_API_KEY": "dummy"},
        )
        assert "LAZY_IMPORT_OK" in p.stdout, (
            f"无 torch 环境下失败:\nstdout={p.stdout}\nstderr={p.stderr[-1200:]}"
        )

    def test_bert_router_still_works_with_heavy_deps(self):
        """惰性化后，真实环境（有 torch）里 bert 路由器仍能正常加载。"""
        from routellm.routers.routers import ROUTER_CLS

        assert "bert" in ROUTER_CLS
        assert "causal_llm" in ROUTER_CLS
        assert "sw_ranking" in ROUTER_CLS
        assert "mf" in ROUTER_CLS
