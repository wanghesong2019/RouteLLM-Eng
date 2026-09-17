"""L1 缓存：进程内 LRU（thread-safe，支持 TTL）。

实现要点：
    - 用 OrderedDict 维护访问顺序，move_to_end 实现 LRU 语义
    - threading.Lock 保护并发访问（网关是多线程/多协程环境）
    - 每条记录带过期时间戳；get 时惰性判断并删除过期项
      （惰性删除足够 —— 缓存项数量受 maxsize 约束，无需后台清扫协程）
    - 命中/未命中计数用于可观测性

命中成本约 0.01ms（纯内存 dict 操作 + 锁），远低于 Redis 网络的 ~1ms。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from routellm.cache.base import CacheBackend


class LRUCache(CacheBackend):
    """进程内 LRU 缓存，带 TTL 与命中统计。

    Args:
        maxsize: 最大条目数，超出时淘汰最久未使用的项。
        default_ttl: 默认存活秒数；None 表示不过期。
    """

    def __init__(self, maxsize: int = 1024, default_ttl: Optional[float] = None):
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.maxsize = int(maxsize)
        self.default_ttl = default_ttl

        # key -> (value, expire_at 或 None)
        self._store: "OrderedDict[str, Tuple[bytes, Optional[float]]]" = OrderedDict()
        self._lock = threading.Lock()

        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[bytes]:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None

            value, expire_at = item
            if expire_at is not None and now >= expire_at:
                # 惰性删除过期项
                del self._store[key]
                self._misses += 1
                return None

            # LRU：标记为最近使用
            self._store.move_to_end(key)
            self._hits += 1
            return value

    async def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
        expire_at = (time.monotonic() + ttl) if ttl is not None else None

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expire_at)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)  # 淘汰最久未使用

    async def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "lru",
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "maxsize": self.maxsize,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        """清空缓存（测试与运维用）。"""
        with self._lock:
            self._store.clear()
