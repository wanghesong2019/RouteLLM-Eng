"""多级缓存编排：L1 → L2 → L3 逐级查找 + 回填。

查找策略：
    依次询问各层；命中则**回填更上层**（下次直接从最快层命中），并返回。

写入策略：
    写入所有层。写失败被忽略（某层故障不应影响整体可用性）。

故障降级：
    任何层的异常都被捕获并跳过，不让单层故障导致缓存不可用
    （如 Redis 挂了，L1 仍应正常工作）。

统计：
    聚合各层统计 + 整体命中率（整体命中率按「跨层查找次数」计，
    而非各层命中数之和 —— 一次请求查了 L1 miss + L2 hit，算 1 hit）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from routellm.cache.base import CacheBackend

logger = logging.getLogger(__name__)


class MultiTierCache:
    """多级缓存，按传入顺序由快到慢查找。

    Args:
        tiers: 缓存后端列表，索引越小越快（如 [LRU, Redis, SQLite]）。

    用法::

        cache = MultiTierCache([LRUCache(maxsize=4096)])
        await cache.set(key, b"...")
        val = await cache.get(key)
    """

    def __init__(self, tiers: List[CacheBackend]):
        if not tiers:
            raise ValueError("至少需要一个缓存层")
        self.tiers = list(tiers)
        self._hits = 0
        self._misses = 0
        # 记录各层命中次数（诊断用，展示哪层在起作用）
        self._tier_hits = [0] * len(self.tiers)

    async def get(self, key: str) -> Optional[bytes]:
        for i, tier in enumerate(self.tiers):
            try:
                val = await tier.get(key)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "缓存层 %d (%s) 读取失败，跳过: %s",
                    i, type(tier).__name__, e,
                )
                continue

            if val is not None:
                self._hits += 1
                self._tier_hits[i] += 1
                # 回填更快的层
                for j in range(i):
                    try:
                        await self.tiers[j].set(key, val)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "回填缓存层 %d 失败: %s", j, e,
                        )
                return val

        self._misses += 1
        return None

    async def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> None:
        for i, tier in enumerate(self.tiers):
            try:
                await tier.set(key, value, ttl)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "缓存层 %d (%s) 写入失败，跳过: %s",
                    i, type(tier).__name__, e,
                )

    async def stats(self) -> Dict[str, Any]:
        tier_stats: List[Dict[str, Any]] = []
        for i, tier in enumerate(self.tiers):
            try:
                s: Dict[str, Any] = dict(await tier.stats())
            except Exception as e:  # noqa: BLE001
                s = {"backend": type(tier).__name__, "error": str(e)}
            s["tier_hits"] = self._tier_hits[i]
            tier_stats.append(s)

        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "tiers": tier_stats,
        }
