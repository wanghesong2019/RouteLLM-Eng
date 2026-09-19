"""指标持久化（SQLite）。

为什么用 SQLite 而非内存：
    方案文档 4.2.1 指出的原始问题就是 `model_counts` 是**全局变量，重启丢失**。
    SQLite 提供持久化，且无需额外组件（对比 Redis/时序库），适合单机网关。

为什么用「写连接 + 读连接」分离：
    FastAPI 是多线程环境，SQLite 连接不能跨线程共享。
    实现上用 `check_same_thread=False` + 写锁保证安全，读操作独立连接。

线程/协程安全：写入用 threading.Lock 串行化，避免 SQLITE_BUSY。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from routellm.monitoring.metrics import RequestMetrics, cost_if_strong, estimate_cost

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT NOT NULL,
    timestamp         REAL NOT NULL,
    prompt_hash       TEXT NOT NULL,
    router_name       TEXT NOT NULL,
    threshold         REAL,
    win_rate          REAL,
    routed_model      TEXT NOT NULL,
    routing_latency_ms REAL,
    llm_latency_ms    REAL,
    total_latency_ms  REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    estimated_cost    REAL,
    cache_hit         INTEGER,
    status            TEXT,
    error_message     TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(routed_model);
"""


class MetricsStore:
    """指标存储（SQLite）。

    Args:
        db_path: SQLite 文件路径。":memory:" 表示内存库（测试用）。
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._lock = threading.Lock()
        # 内存库需保持单一连接（否则每次连接都是新库）
        self._mem_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            # row_factory 必须设置 —— 否则 fetchone() 返回 tuple，
            # 后续 row["col"] 会 TypeError（实测踩过）
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.executescript(_SCHEMA)
            self._mem_conn.commit()

    # ------------------------------------------------------------------ 连接

    def _conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self, c: sqlite3.Connection) -> None:
        if self._mem_conn is None:
            c.executescript(_SCHEMA)
            c.commit()

    # ------------------------------------------------------------------ 写

    async def save(self, m: RequestMetrics) -> None:
        """保存一条请求指标。"""
        await asyncio.to_thread(self._save_sync, m)

    def _save_sync(self, m: RequestMetrics) -> None:
        row = m.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        with self._lock:
            c = self._conn()
            try:
                self._init_schema(c)
                c.execute(
                    f"INSERT INTO requests ({cols}) VALUES ({placeholders})",
                    list(row.values()),
                )
                c.commit()
            except Exception as e:  # noqa: BLE001
                logger.warning("指标写入失败: %s", e)
            finally:
                if self._mem_conn is None:
                    c.close()

    # ------------------------------------------------------------------ 读

    async def summary(self) -> Dict[str, Any]:
        """聚合统计（Dashboard 顶部卡片 + 图表数据）。"""
        return await asyncio.to_thread(self._summary_sync)

    async def window_stats(self, since: float) -> Dict[str, Any]:
        """查询指定时间点以来的聚合统计（供自适应阈值控制器使用）。

        Args:
            since: Unix 时间戳，查询 [since, now] 窗口

        Returns:
            {
                "tokens_per_min": float,     # 窗口内每分钟 Token 消耗
                "latency_p90": float,        # 窗口内 P90 总延迟
                "strong_count": int,         # 窗口内走强模型次数
                "weak_count": int,           # 窗口内走弱模型次数
                "total_count": int,          # 窗口内总请求数
                "total_cost_usd": float,     # 窗口内实际成本
            }
        """
        return await asyncio.to_thread(self._window_stats_sync, since)

    def _window_stats_sync(self, since: float) -> Dict[str, Any]:
        with self._lock:
            c = self._conn()
            try:
                self._init_schema(c)
                cur = c.execute(
                    """
                    SELECT
                        COALESCE(SUM(prompt_tokens), 0) AS pt,
                        COALESCE(SUM(completion_tokens), 0) AS ct,
                        COUNT(*) AS n,
                        SUM(CASE WHEN routed_model = 'strong' THEN 1 ELSE 0 END) AS sn,
                        COALESCE(SUM(estimated_cost), 0) AS cost
                    FROM requests
                    WHERE timestamp >= ?
                    """,
                    (since,),
                )
                row = cur.fetchone()
                pt = row["pt"] or 0
                ct = row["ct"] or 0
                n = row["n"] or 0
                sn = row["sn"] or 0
                cost = row["cost"] or 0.0

                # P90 延迟（窗口内）
                cur = c.execute(
                    "SELECT total_latency_ms AS v FROM requests "
                    "WHERE timestamp >= ? AND total_latency_ms IS NOT NULL "
                    "ORDER BY v",
                    (since,),
                )
                lats = [r["v"] for r in cur.fetchall()]
                p90 = _percentile(lats, 0.90)

                now = time.time()
                minutes = max((now - since) / 60.0, 0.1)
                tokens = pt + ct

                return {
                    "tokens_per_min": tokens / minutes,
                    "latency_p90": p90,
                    "strong_count": int(sn),
                    "weak_count": int(n - sn),
                    "total_count": int(n),
                    "total_cost_usd": round(cost, 6),
                }
            finally:
                if self._mem_conn is None:
                    c.close()

    def _summary_sync(self) -> Dict[str, Any]:
        with self._lock:
            c = self._conn()
            try:
                self._init_schema(c)
                cur = c.execute("SELECT COUNT(*) AS n FROM requests")
                total = cur.fetchone()["n"]
                if total == 0:
                    return _empty_summary()

                # 强/弱模型计数
                cur = c.execute(
                    "SELECT routed_model, COUNT(*) AS n FROM requests GROUP BY routed_model"
                )
                counts = {r["routed_model"]: r["n"] for r in cur.fetchall()}
                strong_n = counts.get("strong", 0)
                weak_n = counts.get("weak", 0)

                # 延迟分位
                cur = c.execute("SELECT total_latency_ms AS v FROM requests ORDER BY v")
                lats = [r["v"] for r in cur.fetchall() if r["v"] is not None]

                # 成本：实际 vs 若全走强模型
                # 注：actual 用**按 token 与档位重算的值**，而非入库时传入的
                # estimated_cost —— 避免调用方口径不一致导致节省额失真
                # （实测踩过：传入值与估算口径不同，节省额被 max(0,·) 截成 0）
                cur = c.execute(
                    "SELECT estimated_cost, prompt_tokens, completion_tokens, "
                    "routed_model, cache_hit, status FROM requests"
                )
                rows = cur.fetchall()
                actual = sum(
                    estimate_cost(
                        r["routed_model"] if r["routed_model"] in ("strong", "weak") else "weak",
                        r["prompt_tokens"] or 0,
                        r["completion_tokens"] or 0,
                    )
                    for r in rows
                )
                if_strong = sum(
                    cost_if_strong(r["prompt_tokens"] or 0, r["completion_tokens"] or 0)
                    for r in rows
                )
                cache_hits = sum(1 for r in rows if r["cache_hit"])

                # 路由延迟（不含 LLM）
                cur = c.execute(
                    "SELECT routing_latency_ms AS v FROM requests "
                    "WHERE routing_latency_ms IS NOT NULL ORDER BY v"
                )
                rls = [r["v"] for r in cur.fetchall()]

                return {
                    "total_requests": total,
                    "strong_count": strong_n,
                    "weak_count": weak_n,
                    "strong_ratio": round(strong_n / total, 4),
                    "actual_cost_usd": round(actual, 6),
                    "cost_if_strong_usd": round(if_strong, 6),
                    "cost_saved_usd": round(max(0.0, if_strong - actual), 6),
                    "cache_hit_rate": round(cache_hits / total, 4),
                    "latency_ms": _percentiles(lats),
                    "routing_latency_ms": _percentiles(rls),
                }
            finally:
                if self._mem_conn is None:
                    c.close()

    async def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """最近 N 条请求（按时间倒序）。"""
        return await asyncio.to_thread(self._recent_sync, limit)

    def _recent_sync(self, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            c = self._conn()
            try:
                self._init_schema(c)
                cols = (
                    "request_id, timestamp, prompt_hash, router_name, threshold, "
                    "win_rate, routed_model, routing_latency_ms, llm_latency_ms, "
                    "total_latency_ms, estimated_cost, cache_hit, status"
                )
                cur = c.execute(
                    f"SELECT {cols} FROM requests ORDER BY timestamp DESC, id DESC LIMIT ?",
                    (int(limit),),
                )
                return [dict(r) for r in cur.fetchall()]
            finally:
                if self._mem_conn is None:
                    c.close()

    async def time_series(self, bucket_seconds: float = 60.0) -> List[Dict[str, Any]]:
        """按时间桶聚合（供折线图）。"""
        return await asyncio.to_thread(self._time_series_sync, bucket_seconds)

    def _time_series_sync(self, bucket_seconds: float) -> List[Dict[str, Any]]:
        with self._lock:
            c = self._conn()
            try:
                self._init_schema(c)
                cur = c.execute(
                    """
                    SELECT CAST(timestamp / ? AS INTEGER) * ? AS bucket,
                           COUNT(*) AS count,
                           SUM(CASE WHEN routed_model = 'strong' THEN 1 ELSE 0 END) AS strong_count,
                           AVG(total_latency_ms) AS avg_latency_ms,
                           SUM(estimated_cost) AS cost
                    FROM requests
                    GROUP BY bucket
                    ORDER BY bucket
                    """,
                    (bucket_seconds, bucket_seconds),
                )
                return [
                    {
                        "bucket": float(r["bucket"]),
                        "count": r["count"],
                        "strong_count": r["strong_count"] or 0,
                        "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 2),
                        "cost": round(r["cost"] or 0.0, 6),
                    }
                    for r in cur.fetchall()
                ]
            finally:
                if self._mem_conn is None:
                    c.close()


# ---------------------------------------------------------------------------


def _percentile(values: List[float], q: float) -> float:
    """算单个分位数（最近秩法）。空列表返回 0.0。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return float(values[idx])


def _percentiles(values: List[float]) -> Dict[str, float]:
    """算 P50/P95/P99/mean。空列表返回全 0。"""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}

    def _p(q: float) -> float:
        if len(values) == 1:
            return float(values[0])
        idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return float(values[idx])

    return {
        "p50": round(_p(0.50), 2),
        "p95": round(_p(0.95), 2),
        "p99": round(_p(0.99), 2),
        "mean": round(sum(values) / len(values), 2),
        "count": len(values),
    }


def _empty_summary() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "strong_count": 0,
        "weak_count": 0,
        "strong_ratio": 0.0,
        "actual_cost_usd": 0.0,
        "cost_if_strong_usd": 0.0,
        "cost_saved_usd": 0.0,
        "cache_hit_rate": 0.0,
        "error_rate": 0.0,
        "latency_ms": _percentiles([]),
        "routing_latency_ms": _percentiles([]),
    }
