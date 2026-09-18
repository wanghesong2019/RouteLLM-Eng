"""指数退避重试。

策略（方案文档 4.3）
    max_attempts = 3      总尝试次数（含首次）
    base_delay   = 1.0s   首次退避
    max_delay    = 10.0s  退避上限
    jitter                随机抖动比例，避免「重试风暴」同时涌向恢复中的上游

为什么只重试**可恢复**异常
--------------------------
重试的前提是「这次失败可能是暂时的」。`BadRequestError`/参数校验类错误
重试 3 次只是把同一个错误犯 3 遍，还会放大上游压力、拖长客户端等待。
故按异常语义分类，默认**不重试**未知异常（保守优于激进）。

异常分类用「类名 + 继承链」而非导入 litellm 的具体异常类：
    1. litellm 各版本的异常类层级会变，硬导入易碎
    2. 本模块要能被单测直接调用，不应强依赖 litellm 是否装了
    3. 类名（RateLimitError / Timeout / BadRequestError）是跨版本稳定的
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")

#: 类名包含这些子串 → 视为可恢复，值得重试
RETRYABLE_NAME_HINTS = (
    "timeout",
    "ratelimit",
    "rate_limit",
    "connection",
    "apiconnection",
    "serviceunavailable",
    "internalservererror",
    "temporarily",
    "overloaded",
)

#: 类名包含这些子串 → 明确不可重试（优先于上面的规则判断）
NON_RETRYABLE_NAME_HINTS = (
    "badrequest",
    "authentication",
    "permission",
    "notfound",
    "unprocessable",
    "invalidrequest",
)


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    判定顺序：
      1. 明确的客户端错误（BadRequest/Auth/Permission/NotFound/...）→ 否
      2. 内建的可恢复异常（TimeoutError / ConnectionError / OSError 子集）→ 是
      3. 类名含可恢复关键字（RateLimit/Timeout/Connection/5xx）→ 是
      4. 其余 → 否（保守，避免放大故障）
    """
    # 按 MRO 收集所有类名，覆盖异常层级中的任意一层命中
    names = [cls.__name__.lower() for cls in type(exc).__mro__]

    for n in names:
        if any(h in n for h in NON_RETRYABLE_NAME_HINTS):
            return False

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    for n in names:
        if any(h in n for h in RETRYABLE_NAME_HINTS):
            return True

    return False


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    on_retry: Optional[Callable[[int, BaseException, float], Any]] = None,
) -> T:
    """带指数退避的重试执行器。

    Args:
        fn: 无参 awaitable 工厂（用 lambda/闭包捕获参数）。
        max_attempts: 总尝试次数（含首次）。1 表示不重试。
        base_delay: 首次退避秒数，之后按 2^n 增长。
        max_delay: 单次退避上限。
        jitter: 抖动比例（0~1）。实际延迟 = delay * (1 ± jitter*rand)。
            测试中传 0 以获得确定性。
        on_retry: 可选回调 (attempt, exc, delay)，用于打日志/记指标。

    Returns:
        fn 的返回值。

    Raises:
        最后一次的异常。不可重试异常立即抛出，不做后续尝试。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc

            # 不可重试 或 已是最后一次 → 直接抛出
            if not is_retryable(exc) or attempt >= max_attempts:
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay = delay * (1 + random.uniform(-jitter, jitter))

            if on_retry is not None:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:  # noqa: BLE001
                    pass  # 回调失败不应影响重试主流程

            await asyncio.sleep(delay)

    # 理论不可达（最后一轮必 raise），保留以安抚静态检查
    assert last_exc is not None
    raise last_exc
