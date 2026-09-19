"""fast_path 短文本误判修复的回归测试。

线上实测（2026-09-19）：`证明下pi是无理数`（9 字符）命中「极短文本 ≤15」
规则被判 weak 并跳过 BERT —— 但它是一道正经数学证明题，短 ≠ 简单。

代价不对称：误判跳过 BERT 后，该类问题永远拿不到精准打分（质量问题）；
而多走一次 BERT 只花 10~30ms。因此修复必须**偏保守**：宁可少命中。
"""

from routellm.routers.fast_path import FastPathConfig, FastPathRouter


class TestImperativeQuestionsNotTreatedAsTrivial:
    """含指令性动词的短提问不得被当寒暄。"""

    def test_short_math_proof_not_hit(self):
        fp = FastPathRouter()
        assert fp.evaluate("证明下pi是无理数") is None
        assert fp.evaluate("证明根号2是无理数") is None

    def test_short_imperative_questions_not_hit(self):
        fp = FastPathRouter()
        for q in [
            "解释下傅里叶变换",
            "推导一下欧拉公式",
            "求导一下e的x次方",
            "分析这段代码的问题",
            "计算一下矩阵的行列式",
            "实现一个快排",
            "优化这条SQL语句",
            "比较一下两种方案",
            "写一个正则匹配邮箱",
            "翻译这句话成英文",
        ]:
            assert fp.evaluate(q) is None, f"短提问被误判为寒暄: {q}"

    def test_english_imperatives_not_hit(self):
        fp = FastPathRouter()
        for q in [
            "explain PCA",
            "prove sqrt(2)",
            "write a sort function",
            "analyze this bug",
            "compute the inverse",
        ]:
            assert fp.evaluate(q) is None, f"短英文提问被误判: {q}"


class TestGreetingsStillHit:
    """修复不得把真正的寒暄也挡掉（否则快速通道等于失效）。"""

    def test_greetings_still_hit(self):
        fp = FastPathRouter()
        for g in ["你好", "hi", "hello", "嗨", "在吗", "谢谢", "ok", "好的", "继续", "收到"]:
            assert fp.evaluate(g) is not None, f"寒暄未命中: {g}"

    def test_empty_still_hits(self):
        fp = FastPathRouter()
        assert fp.evaluate("") is not None
        assert fp.evaluate("   ") is not None

    def test_trailing_punctuation_hit(self):
        fp = FastPathRouter()
        assert fp.evaluate("你好！") is not None
        assert fp.evaluate("好的。") is not None


class TestThresholdTightened:
    """阈值默认收紧：15 太宽（"解释下傅里叶变换"才 8 字符，本就该走 BERT）。"""

    def test_default_threshold_is_tighter(self):
        assert FastPathConfig().short_text_threshold <= 8, (
            "默认阈值仍过宽 —— 8 字以内的正经提问（如『解释下傅里叶变换』）"
            "会被当成寒暄跳过 BERT"
        )

    def test_configured_threshold_still_honored(self):
        """显式配置的阈值仍应生效（热更新/环境变量可控）。"""
        fp = FastPathRouter(FastPathConfig(short_text_threshold=3))
        assert fp.evaluate("你好") is not None      # 2 字符 ≤ 3 → 命中
        # 4 字符 > 3 → 不因长度命中；也不匹配寒暄正则 → 应放行
        assert fp.evaluate("这四个字") is None
        assert fp.evaluate("分析问题") is None

    def test_disable_still_works(self):
        fp = FastPathRouter(FastPathConfig(enabled=False))
        assert fp.evaluate("你好") is None
