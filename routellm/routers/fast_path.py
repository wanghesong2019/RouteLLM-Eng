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
    # 极短文本阈值：字符数 ≤ 此值视为极简（寒暄/确认类）
    short_text_threshold: int = 15
    # 正则模式列表：命中即直通弱模型
    patterns: tuple = (
        r"^(你好|hi|hello|hey|嗨|在吗|在不在)[\s!！。.]*$",
        r"^(好的|明白|了解|收到|ok|OK|嗯|嗯嗯)[\s!！。.]*$",
        r"^(继续|continue|go on)[\s!！。.]*$",
        r"^(谢谢|感谢|thanks|thank you|多谢)[\s!！。.]*$",
        r"^(是的|对的|没错|对|yes|no|不是)[\s!！。.]*$",
    )


class FastPathRouter:
    """L1 快速通道路由器。

    纯规则匹配，不调用任何模型。
    命中 → ("weak", 0.0, "fast_path")；未命中 → None（放行给 BERT）
    """

    def __init__(self, config: Optional[FastPathConfig] = None):
        self.config = config or FastPathConfig()
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.config.patterns]

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

        # 规则 1：极短文本
        if len(text) <= self.config.short_text_threshold:
            return ("weak", 0.0, "fast_path")

        # 规则 2：正则匹配（寒暄/确认类固定话术）
        for pat in self._compiled:
            if pat.search(text):
                return ("weak", 0.0, "fast_path")

        return None

    def update_config(self, config: FastPathConfig) -> None:
        """热更新配置（重新编译正则）。"""
        self.config = config
        self._compiled = [re.compile(p, re.IGNORECASE) for p in config.patterns]
