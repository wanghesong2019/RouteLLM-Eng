"""多级缓存层。

设计依据：改造方案 4.1 节 —— 方案 C（LRU + 可选 Redis + 可选 SQLite）。

针对本项目的现实调整
--------------------
方案文档写作时的瓶颈是 OpenAI Embedding API（~50ms/次），故缓存对象是
**embedding 向量**。但实测瓶颈已变为 Elo 回归（`LogisticRegression.fit`
占路由延迟 90%，355ms），因此本实现**优先缓存 win_rate 结果** ——
命中时连相似度计算与 Elo 回归都跳过，这才是降延迟的关键。

缓存分两类 key（见 keys.py）：
    result:    prompt → win_rate（省掉整条推理链路）
    embedding: prompt → embedding 向量（供需要向量的场景复用）

分层：
    L1 LRUCache     进程内，命中 ~0.01ms，容量受限，重启丢失
    L2/L3 可选      由调用方注入（Redis / SQLite 等），本模块不做假设

任一层失败不影响整体可用性（降级为跳过该层）。
"""

from routellm.cache.base import CacheBackend
from routellm.cache.keys import embedding_key, result_key
from routellm.cache.lru_cache import LRUCache
from routellm.cache.multi_tier import MultiTierCache

__all__ = [
    "CacheBackend",
    "LRUCache",
    "MultiTierCache",
    "result_key",
    "embedding_key",
]
