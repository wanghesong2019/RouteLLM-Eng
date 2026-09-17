"""缓存接入 SWRankingRouter 的单元测试（TDD RED 阶段）。

接入点（方案文档 4.1.4）：
    calculate_strong_win_rate 中先查缓存，命中直接返回；
    未命中则计算并回写。

关键设计（针对本项目现实）：
    缓存 **win_rate 结果**（而非仅 embedding）—— 因为实测瓶颈是
    Elo 回归（355ms，占 90%），缓存结果才能命中时省掉整条链路。

    同理 host 服务的 score_batch 也接入同一套缓存。

一致性要求：
    缓存命中返回的值必须与未缓存计算的值**完全一致**（浮点位级），
    否则缓存会引入路由决策的不确定性。

运行：
    pytest tests/test_router_cache.py -v
"""

import asyncio
import json
import os

import numpy as np
import pandas as pd
import pytest


def _make_battles(n_rows=200, seed=0):
    models = ["gpt-4-0613", "claude-v1", "mixtral-8x7b-instruct-v0.1", "gpt-4-1106-preview"]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        rows.append({
            "id": f"t-{i}",
            "model_a": models[i % len(models)],
            "model_b": models[(i + 1) % len(models)],
            "prompt": json.dumps([f"Prompt {i} padded beyond sixteen chars.", "follow"]),
            "winner_model_a": 1 if i % 3 == 0 else 0,
            "winner_model_b": 1 if i % 3 == 1 else 0,
            "winner_tie": 1 if i % 3 == 2 else 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def local_data(tmp_path):
    """battles csv + embeddings npy（条数一致）。"""
    n = 200
    csv = tmp_path / "b.csv"
    _make_battles(n).to_csv(csv, index=False)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, 1024)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    npy = tmp_path / "e.npy"
    np.save(npy, v)
    return str(csv), str(npy)


@pytest.fixture
def router(local_data, monkeypatch):
    """配好假编码器的 router —— 避免测试依赖真实 bge-m3 权重或 OpenAI key。

    编码器的行为对缓存测试不重要（缓存的对象是 win_rate 结果），
    只需保证它是确定性的：同一 prompt → 同一向量。
    """
    from routellm.routers.routers import SWRankingRouter

    csv, npy = local_data
    r = SWRankingRouter(local_battles_csv=csv, local_embeddings_npy=npy)

    _vec_cache = {}

    def fake_encode(prompt):
        if prompt not in _vec_cache:
            seed = abs(hash(prompt)) % (2**31)
            g = np.random.default_rng(seed)
            v = g.standard_normal(1024).astype(np.float32)
            _vec_cache[prompt] = v / np.linalg.norm(v)
        return _vec_cache[prompt]

    monkeypatch.setattr(r, "_encode_prompt", fake_encode)
    return r


# ------------------------------------------------- 缓存命中行为


def test_cache_hit_skips_computation(router, monkeypatch):
    """缓存命中时不应重复执行推理（用 spy 计数验证）。"""
    from routellm.cache import LRUCache, MultiTierCache

    router.cache = MultiTierCache([LRUCache(maxsize=100)])

    calls = {"n": 0}
    orig = router._encode_prompt

    def spy(prompt):
        calls["n"] += 1
        return orig(prompt)

    monkeypatch.setattr(router, "_encode_prompt", spy)

    p = "What is the capital of France?"
    r1 = router.calculate_strong_win_rate(p)
    n_after_first = calls["n"]
    r2 = router.calculate_strong_win_rate(p)  # 应命中缓存

    assert calls["n"] == n_after_first, "第二次不应重新编码（缓存命中）"
    assert r1 == r2, "缓存结果应完全一致"


def test_cache_miss_then_populate(router):
    """未命中后应回写缓存，后续命中。"""
    from routellm.cache import LRUCache, MultiTierCache
    from routellm.routers.routers import SWRankingRouter

    cache = MultiTierCache([LRUCache(maxsize=100)])
    router.cache = cache

    p = "Explain the trade-offs between consistency and availability."
    router.calculate_strong_win_rate(p)

    s = asyncio.run(cache.stats())
    assert s["misses"] == 1, "首次应 miss"
    assert s["hits"] == 0

    router.calculate_strong_win_rate(p)
    s = asyncio.run(cache.stats())
    assert s["hits"] == 1, "第二次应 hit"


def test_cached_value_bitwise_identical(router):
    """缓存值与直接计算值须位级一致（避免引入决策不确定性）。"""
    from routellm.cache import LRUCache, MultiTierCache
    from routellm.routers.routers import SWRankingRouter

    router.cache = MultiTierCache([LRUCache(maxsize=100)])

    p = "Prove that the square root of 2 is irrational."
    first = router.calculate_strong_win_rate(p)  # 计算
    second = router.calculate_strong_win_rate(p)  # 缓存
    assert first == second


def test_different_prompts_not_collide(router):
    """不同 prompt 不应共享缓存项。"""
    from routellm.cache import LRUCache, MultiTierCache
    from routellm.routers.routers import SWRankingRouter

    cache = MultiTierCache([LRUCache(maxsize=100)])
    router.cache = cache

    router.calculate_strong_win_rate("prompt AAAA long enough text")
    router.calculate_strong_win_rate("prompt BBBB long enough text")

    s = asyncio.run(cache.stats())
    assert s["misses"] == 2, "两个不同 prompt 应各 miss 一次"


# ------------------------------------------------- 无缓存时行为不变


def test_no_cache_still_works(router):
    """未配置 cache 时行为与传统一致（向后兼容）。"""
    assert getattr(router, "cache", None) is None
    wr = router.calculate_strong_win_rate("hello world test prompt here")
    assert 0.0 <= wr <= 1.0


def test_cache_stats_exposed_on_router(router):
    """router 应暴露缓存统计（供 /health 汇总）。"""
    from routellm.cache import LRUCache, MultiTierCache

    router.cache = MultiTierCache([LRUCache(maxsize=100)])
    router.calculate_strong_win_rate("a prompt long enough to pass filters")

    s = asyncio.run(router.cache_stats())
    assert "hit_rate" in s
    assert "tiers" in s


def test_cache_failure_degrades_gracefully(router):
    """缓存故障时路由仍应正常返回（降级）。"""
    from routellm.cache.base import CacheBackend
    from routellm.cache.multi_tier import MultiTierCache

    class Broken(CacheBackend):
        async def get(self, key):
            raise ConnectionError("boom")

        async def set(self, key, value, ttl=None):
            raise ConnectionError("boom")

        async def stats(self):
            raise ConnectionError("boom")

    router.cache = MultiTierCache([Broken()])

    wr = router.calculate_strong_win_rate("prompt that is long enough here")
    assert 0.0 <= wr <= 1.0, "缓存全挂时仍应返回结果"
