"""指标数据模型与成本估算。

设计依据：方案文档 4.2.4 节的指标体系。

隐私约定：**只存 prompt 的 hash，不存明文**。这既符合隐私要求，
又能在排查时关联同一 prompt 的多次请求（用于验证缓存命中）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 成本估算
# ---------------------------------------------------------------------------
# 单价（美元 / 1M token）。取自公开定价的近似值，仅用于「成本节省」的
# 量级展示，非精确账单。真实计费以 provider 侧为准。
PRICE_PER_MTOK = {
    "strong": {"prompt": 10.0, "completion": 30.0},   # 类比 GPT-4 级
    "weak": {"prompt": 0.25, "completion": 0.25},     # 类比 Mixtral 级
}


def estimate_cost(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按 token 数与模型档位估算成本（USD）。

    Args:
        tier: "strong" 或 "weak"
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数

    Returns:
        估算成本（USD）。未知档位按 weak 计（保守）。
    """
    price = PRICE_PER_MTOK.get(tier, PRICE_PER_MTOK["weak"])
    return (prompt_tokens * price["prompt"] + completion_tokens * price["completion"]) / 1_000_000


def cost_if_strong(prompt_tokens: int, completion_tokens: int) -> float:
    """若该请求走强模型会花多少（用于算节省额）。"""
    return estimate_cost("strong", prompt_tokens, completion_tokens)


def hash_prompt(text: str) -> str:
    """prompt 的 hash（隐私：指标只存 hash，不存明文）。

    与缓存 key 用同一算法（sha256 前 16 位），便于关联「同一 prompt 的
    缓存命中」与「指标记录」。
    """
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 请求级指标
# ---------------------------------------------------------------------------
@dataclass
class RequestMetrics:
    """单个请求的路由与成本指标。

    字段对应方案文档 4.2.4 的指标体系。
    """

    request_id: str
    timestamp: float
    prompt_hash: str          # 隐私保护：只存 hash，不存明文
    router_name: str
    threshold: float
    win_rate: float           # 路由置信度
    routed_model: str         # "strong" / "weak"
    routing_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    cache_hit: bool
    status: str               # "success" / "error"
    error_message: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        """转为 SQLite 可写入的 dict（bool → int）。"""
        d = asdict(self)
        d["cache_hit"] = 1 if self.cache_hit else 0
        return d
