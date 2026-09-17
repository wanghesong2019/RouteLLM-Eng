"""多级缓存层的单元测试（TDD RED 阶段）。

设计依据：docs（方案文档 4.1 节）—— 方案 C 多级缓存
    L1: 进程内 LRU（命中 ~0.01ms）
    L2: Redis（命中 ~1ms，可选）
    L3: SQLite 磁盘持久化（冷启动恢复）

针对本项目的现实调整：
    方案文档写作时瓶颈是 OpenAI Embedding API（50ms），故缓存对象是
    **embedding 向量**。但实测瓶颈已变为 Elo 回归（355ms，占路由延迟 90%），
    因此本实现**优先缓存 win_rate 结果** —— 命中时连相似度计算与
    Elo 回归都跳过，这才是降延迟的关键。

    缓存分两类 key：
      - 结果缓存: prompt → win_rate（省掉整条推理链路）
      - 向量缓存: prompt → embedding（供需要向量的场景复用）

本测试聚焦 L1 + 多级编排，不依赖 Redis（L2 为可选组件）。

运行：
    pytest tests/test_cache.py -v
"""

import asyncio
import time

import pytest


# --------------------------------------------------------- L1 LRU 基础行为


def test_lru_basic_set_get():
    """L1 LRU 基本存取。"""
    from routellm.cache.lru_cache import LRUCache

    c = LRUCache(maxsize=10)
    assert asyncio.run(c.get("k")) is None
    asyncio.run(c.set("k", b"v"))
    assert asyncio.run(c.get("k")) == b"v"


def test_lru_eviction_on_maxsize():
    """超出容量应淘汰最久未使用的项。"""
    from routellm.cache.lru_cache import LRUCache

    c = LRUCache(maxsize=3)
    for i in range(3):
        asyncio.run(c.set(f"k{i}", f"v{i}".encode()))
    # 访问 k0，使 k1 成为最久未使用
    asyncio.run(c.get("k0"))
    asyncio.run(c.set("k3", b"v3"))

    assert asyncio.run(c.get("k1")) is None, "k1 应被淘汰"
    assert asyncio.run(c.get("k0")) == b"v0", "k0 刚被访问，应保留"
    assert asyncio.run(c.get("k3")) == b"v3"


def test_lru_ttl_expiry():
    """带 TTL 的项过期后应取不到。"""
    from routellm.cache.lru_cache import LRUCache

    c = LRUCache(maxsize=10, default_ttl=0.2)
    asyncio.run(c.set("k", b"v"))
    assert asyncio.run(c.get("k")) == b"v"
    time.sleep(0.25)
    assert asyncio.run(c.get("k")) is None, "TTL 过期后应 miss"


def test_lru_stats_tracks_hits_misses():
    """统计应记录命中/未命中次数。"""
    from routellm.cache.lru_cache import LRUCache

    c = LRUCache(maxsize=10)
    asyncio.run(c.set("k", b"v"))
    asyncio.run(c.get("k"))  # hit
    asyncio.run(c.get("missing"))  # miss

    s = asyncio.run(c.stats())
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1


# --------------------------------------------------------- 多级编排


def test_multi_tier_l1_hit_skips_lower():
    """L1 命中时不应访问下层。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    class SpyCache(LRUCache):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.get_calls = 0

        async def get(self, key):
            self.get_calls += 1
            return await super().get(key)

    l1 = LRUCache(maxsize=10)
    l2 = SpyCache(maxsize=10)
    c = MultiTierCache([l1, l2])

    asyncio.run(l1.set("k", b"v"))
    assert asyncio.run(c.get("k")) == b"v"
    assert l2.get_calls == 0, "L1 命中不应查 L2"


def test_multi_tier_backfills_upper_on_lower_hit():
    """下层命中应回填上层（下次直接从上层命中）。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    l1 = LRUCache(maxsize=10)
    l2 = LRUCache(maxsize=10)
    c = MultiTierCache([l1, l2])

    asyncio.run(l2.set("k", b"v"))  # 只在下层
    assert asyncio.run(c.get("k")) == b"v"
    # 回填后 L1 应有
    assert asyncio.run(l1.get("k")) == b"v", "下层命中后应回填 L1"


def test_multi_tier_all_miss_returns_none():
    """全层未命中返回 None。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    c = MultiTierCache([LRUCache(maxsize=10), LRUCache(maxsize=10)])
    assert asyncio.run(c.get("nope")) is None


def test_multi_tier_set_writes_all_layers():
    """写入应落所有层。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    l1, l2 = LRUCache(maxsize=10), LRUCache(maxsize=10)
    c = MultiTierCache([l1, l2])
    asyncio.run(c.set("k", b"v"))
    assert asyncio.run(l1.get("k")) == b"v"
    assert asyncio.run(l2.get("k")) == b"v"


def test_multi_tier_aggregated_stats():
    """统计应含各层信息与整体命中率。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    c = MultiTierCache([LRUCache(maxsize=10), LRUCache(maxsize=10)])
    asyncio.run(c.set("k", b"v"))
    asyncio.run(c.get("k"))  # hit
    asyncio.run(c.get("miss"))  # miss

    s = asyncio.run(c.stats())
    assert "tiers" in s
    assert len(s["tiers"]) == 2
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(0.5)


def test_multi_tier_survives_lower_tier_failure():
    """某层故障不应导致整个缓存不可用（降级为跳过该层）。"""
    from routellm.cache.lru_cache import LRUCache
    from routellm.cache.multi_tier import MultiTierCache

    class BrokenCache(LRUCache):
        async def get(self, key):
            raise ConnectionError("模拟 Redis 不可达")

        async def set(self, key, value, ttl=None):
            raise ConnectionError("模拟 Redis 不可达")

    l1 = LRUCache(maxsize=10)
    broken = BrokenCache(maxsize=10)
    c = MultiTierCache([l1, broken])

    # 写入不应崩（broken 层失败被忽略）
    asyncio.run(c.set("k", b"v"))
    assert asyncio.run(l1.get("k")) == b"v", "健康层应仍写入成功"
    # 读取应能命中健康层
    assert asyncio.run(c.get("k")) == b"v"


# --------------------------------------------------------- key 生成


def test_key_generation_deterministic_and_distinct():
    """相同 prompt 生成相同 key；不同 prompt 生成不同 key。"""
    from routellm.cache.keys import result_key, embedding_key

    assert result_key("hello") == result_key("hello")
    assert result_key("hello") != result_key("world")
    assert embedding_key("hello") == embedding_key("hello")
    assert embedding_key("hello") != result_key("hello"), "两类 key 前缀应不同"


def test_key_is_short():
    """key 应短（用 hash 而非原文），避免缓存膨胀。"""
    from routellm.cache.keys import result_key

    long_prompt = "x" * 10000
    k = result_key(long_prompt)
    assert len(k) < 64, f"key 过长: {len(k)}"
    assert long_prompt not in k
