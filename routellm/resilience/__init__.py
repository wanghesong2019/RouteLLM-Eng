"""容错与 Resilience（方案文档 4.3）。

针对「问题3：无容错机制——单点故障即全盘崩溃」。

三个组件：
    circuit_breaker  三态熔断器（CLOSED → OPEN → HALF_OPEN → CLOSED）
    retry            指数退避重试，只重试可恢复异常
    fallback         降级策略链：强 → 弱 → 缓存 → 503

设计原则：本包不 import litellm（异常按类名语义判定），因此可被单测
直接调用，也不受 litellm 版本异常层级变化影响。
"""

from routellm.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from routellm.resilience.fallback import (
    FallbackExhaustedError,
    FallbackResult,
    ResilientCaller,
)
from routellm.resilience.retry import is_retryable, with_retry

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "FallbackExhaustedError",
    "FallbackResult",
    "ResilientCaller",
    "is_retryable",
    "with_retry",
]
