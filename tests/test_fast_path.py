"""FastPathRouter 测试（方案文档 2.2）。"""

from routellm.routers.fast_path import FastPathConfig, FastPathRouter


class TestFastPath:
    def test_short_text_hits(self):
        """极短文本应命中快速通道。"""
        fp = FastPathRouter()
        result = fp.evaluate("你好")
        assert result is not None
        assert result[0] == "weak"
        assert result[2] == "fast_path"

    def test_greeting_pattern_hits(self):
        """寒暄模式应命中。"""
        fp = FastPathRouter()
        assert fp.evaluate("hello!") is not None
        assert fp.evaluate("好的。") is not None
        assert fp.evaluate("继续") is not None

    def test_complex_query_misses(self):
        """复杂 Query 不应命中。"""
        fp = FastPathRouter()
        assert (
            fp.evaluate("请帮我分析一下 Transformer 架构中多头注意力机制的数学原理")
            is None
        )
        assert fp.evaluate("写一个 Python 函数实现快速排序") is None

    def test_disabled_returns_none(self):
        """禁用时应返回 None。"""
        fp = FastPathRouter(FastPathConfig(enabled=False))
        assert fp.evaluate("你好") is None

    def test_empty_prompt_hits(self):
        """空 prompt 应命中（走弱模型，不浪费 BERT）。"""
        fp = FastPathRouter()
        assert fp.evaluate("") is not None
        assert fp.evaluate("   ") is not None

    def test_update_config_recompiles_patterns(self):
        """热更新配置后正则应重新编译生效（短文本规则同时调小以隔离正则维度）。"""
        fp = FastPathRouter()
        assert fp.evaluate("你好") is not None
        fp.update_config(FastPathConfig(short_text_threshold=1, patterns=(r"^zzz$",)))
        assert fp.evaluate("你好") is None     # 正则已换，且非极短文本
        assert fp.evaluate("zzz") is not None  # 新正则生效

    def test_threshold_configurable(self):
        """极短文本阈值可通过配置调整。"""
        fp = FastPathRouter(FastPathConfig(short_text_threshold=3))
        assert fp.evaluate("你好") is not None
        # 4 个字符 且 不匹配任何寒暄正则 → 未命中
        assert fp.evaluate("分析一下") is None
