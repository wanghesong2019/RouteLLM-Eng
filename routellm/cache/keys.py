"""缓存 key 生成。

key 用 hash 而非 prompt 原文，原因：
    1. prompt 可达数 KB，直接作 key 会让缓存体积膨胀
    2. Redis/磁盘后端的 key 长度有限制
    3. 避免缓存中留存用户原文（隐私友好）

两类 key 区分前缀，避免「结果」与「向量」互相污染：
    routellm:res:<sha256前16位>
    routellm:emb:<sha256前16位>
"""

from __future__ import annotations

import hashlib

RESULT_PREFIX = "routellm:res:"
EMBEDDING_PREFIX = "routellm:emb:"

# hash 截断长度。16 位十六进制 = 64 bit，在 prompt 规模（百万级）下
# 碰撞概率可忽略，且 key 足够短。
_HASH_LEN = 16


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LEN]


def result_key(prompt: str) -> str:
    """win_rate 结果缓存的 key。"""
    return f"{RESULT_PREFIX}{_hash(prompt)}"


def embedding_key(prompt: str) -> str:
    """embedding 向量缓存的 key。"""
    return f"{EMBEDDING_PREFIX}{_hash(prompt)}"
