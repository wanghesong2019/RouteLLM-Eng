"""熔断器：三态状态机 CLOSED → OPEN → HALF_OPEN → CLOSED。

职责边界（重要设计决策）
------------------------
熔断器**只回答「现在能不能放行」**，不负责调用、不抛业务异常、不做降级。
调用方拿到 False 后自行决定降级策略。这样熔断器可以独立测试，也不会
把「基础设施状态」和「业务错误」混在一个异常体系里。

参数（方案文档 4.3）
    failure_threshold   连续失败多少次开路（默认 5）
    recovery_timeout    开路后多久转半开（秒，默认 60）
    half_open_max_calls 半开态最多放行多少次探测（默认 3）

为什么记录「连续失败」而非「累计失败」：累计计数会让一个长期运行、
偶发失败的服务最终必然开路（失败总数只增不减）。连续失败配合
成功清零，才表达「当前是否持续故障」。
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Callable, Optional


class CircuitState(enum.Enum):
    """熔断器三态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断器处于开路状态，拒绝请求。

    作为异常提供，便于需要「直接抛错」风格调用方使用；
    `CircuitBreaker.allow_request()` 走 bool 路径，两者并存。
    """


class CircuitBreaker:
    """线程安全的三态熔断器。

    用锁保护状态转移：网关是多协程并发，但状态转移本身是极短临界区，
    用 threading.Lock 足够且不引入 async 传染（熔断器可能被同步代码调用）。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock or time.monotonic

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """当前状态（会先根据时间推进 HALF_OPEN 转换）。"""
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def _maybe_half_open(self) -> None:
        """开路满恢复期 → 转半开。调用方需持锁。"""
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0

    def allow_request(self) -> bool:
        """当前是否允许放行一次请求。"""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False  # OPEN

    def record_success(self) -> None:
        """记录一次成功。半开态下成功 → 闭合；闭合态下清零连续失败。"""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._half_open_calls = 0
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """记录一次失败。达到阈值 → 开路；半开态下失败 → 立即开路。"""
        with self._lock:
            self._consecutive_failures += 1
            if self._state is CircuitState.HALF_OPEN:
                # 探测失败：立即重新开路，重新计时
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_calls = 0
                return
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_calls = 0

    def reset(self) -> None:
        """强制回到闭合态（运维手动恢复用）。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._half_open_calls = 0
