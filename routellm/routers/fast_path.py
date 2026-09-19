"""级联前置过滤：L1 快速通道（方案文档 2.2）。

在 BERT 路由之前拦截确定性简单 Query，直接判定走弱模型。
只有非平凡 Query 才放行给 BERT 精准打分。

设计原则：
- 纯规则，零模型调用，延迟 <1ms
- 命中返回 ("weak", 0.0, "fast_path")；未命中返回 None
- 不命中时不产生任何副作用（不写缓存、不改状态）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# 返回值：(routed_tier, win_rate, router_name)
# routed_tier: "weak" 表示直通弱模型
FastPathResult = Optional[Tuple[str, float, str]]


@dataclass
class FastPathConfig:
    """快速通道配置（后续可通过 ConfigStore 热更新）。"""

    enabled: bool = True
    # 极短文本阈值：字符数 ≤ 此值视为极简（寒暄/确认类）。
    #
    # 为什么是 8 而不是方案文档写的 15：线上实测「解释下傅里叶变换」只有 8 个
    # 字符、「证明下pi是无理数」9 个字符 —— 都是正经提问，却会被 ≤15 的规则
    # 当寒暄处理并跳过 BERT。误判代价不对称：跳过 BERT 后这类问题永远拿不到
    # 精准打分（质量问题），而多走一次 BERT 只花 10~30ms。故收紧到 8，并配合
    # imperative_patterns 做二次排除。
    short_text_threshold: int = 8
    # 正则模式列表：命中即直通弱模型（寒暄/确认类固定话术）
    patterns: tuple = (
        r"^(你好|hi|hello|hey|嗨|在吗|在不在)[\s!！。.]*$",
        r"^(好的|明白|了解|收到|ok|OK|嗯|嗯嗯)[\s!！。.]*$",
        r"^(继续|continue|go on)[\s!！。.]*$",
        r"^(谢谢|感谢|thanks|thank you|多谢)[\s!！。.]*$",
        r"^(是的|对的|没错|对|yes|no|不是)[\s!！。.]*$",
    )
    # 指令性模式：短文本命中**这些**则不视为极简，放行给 BERT。
    #
    # 「短」不等于「简单」—— 数学证明、推导、代码题往往很短但难度极高。
    # 一旦被当寒暄降级，就永远绕过 BERT 打分了。这里是风险控制的兜底：
    # 命中即可疑，宁可多付 10~30ms 的 BERT RPC，也不误伤正经提问。
    imperative_patterns: tuple = (
        # 中文指令性动词
        r"(证明|推导|求证|推导出|解析|解释|说明|阐述|论述|分析|辨析|比较|对比|"
        r"计算|求解|求|求导|积分|化简|展开|估算|实现|编写|写一?个?|重构|优化|"
        r"调试|排查|定位|修复|设计|架构|迁移|转换|翻译|总结|归纳|提取|生成|"
        r"列出|列举|生成|检查|审查|评审|评估|判断|论证|举例)",
        # 英文指令性动词
        r"\b(prove|derive|explain|analyze|analyse|compare|contrast|compute|"
        r"calculate|solve|implement|write|refactor|optimize|optimise|debug|"
        r"design|migrate|translate|summarize|summarise|extract|generate|"
        r"list|review|evaluate|argue|justify)\b",
    )


class FastPathRouter:
    """L1 快速通道路由器。

    纯规则匹配，不调用任何模型。
    命中 → ("weak", 0.0, "fast_path")；未命中 → None（放行给 BERT）
    """

    def __init__(self, config: Optional[FastPathConfig] = None):
        self.config = config or FastPathConfig()
        self._compile()

    def _compile(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.config.patterns]
        self._compiled_imperative = [
            re.compile(p, re.IGNORECASE) for p in self.config.imperative_patterns
        ]

    def _is_imperative(self, text: str) -> bool:
        """文本是否含指令性表述（短但不简单）。"""
        return any(p.search(text) for p in self._compiled_imperative)

    def evaluate(self, prompt: str) -> FastPathResult:
        """评估 prompt 是否命中快速通道。

        Args:
            prompt: 用户最后一条消息的文本

        Returns:
            ("weak", 0.0, "fast_path") 如果命中；None 如果未命中
        """
        if not self.config.enabled:
            return None

        text = (prompt or "").strip()
        if not text:
            # 空 prompt 无信息量，浪费一次 BERT RPC 毫无意义
            return ("weak", 0.0, "fast_path")

        # 规则 1：极短文本 —— 但含指令性表述的除外（短 ≠ 简单）
        if len(text) <= self.config.short_text_threshold:
            if not self._is_imperative(text):
                return ("weak", 0.0, "fast_path")

        # 规则 2：正则匹配（寒暄/确认类固定话术）
        for pat in self._compiled:
            if pat.search(text):
                return ("weak", 0.0, "fast_path")

        return None

    def update_config(self, config: FastPathConfig) -> None:
        """热更新配置（重新编译正则）。"""
        self.config = config
        self._compile()
