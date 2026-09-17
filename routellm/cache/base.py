"""缓存后端抽象基类。"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class CacheBackend(abc.ABC):
    """缓存后端接口。

    所有方法均为异步 —— 即便 L1 是同步的内存操作，也统一为 async，
    以便上层 MultiTierCache 用同一套代码编排同步/异步混合的后端
    （如 Redis 客户端天然异步）。

    实现约定：
        - get 未命中返回 None（不抛异常）
        - set 的 ttl 为秒；None 表示用后端默认值
        - 后端自身故障由上层捕获并降级，实现可自由抛出
    """

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """按键取值。未命中返回 None。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> None:
        """写入键值。ttl 秒；None 用后端默认。"""
        raise NotImplementedError

    @abc.abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """返回统计信息（至少含 hits / misses / size）。"""
        raise NotImplementedError
