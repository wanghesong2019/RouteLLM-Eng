"""自适应阈值闭环控制器（方案文档 3.2）。

将 RouteLLM 的静态阈值 τ 升级为动态阈值 τ(t)，
根据实时成本/延迟指标在 [τ_min, τ_max] 区间内平滑调节。

核心公式：
    τ(t) = clip(τ_base + K_p × (cost_rate - budget_rate)/budget_rate, τ_min, τ_max)
           （延迟维度权重减半叠加）

硬质量防线（Hard Ceiling Guardrail）：
    - s ≥ τ_max          → 强制走 Strong，无论预算多紧张/τ 被压多低
    - s < τ_min          → 稳定走 Weak（安全降本区）
    - 预算耗尽 + s ≥ τ_max → HTTP 429 背压（不以次充好，绝不静默降级）

设计依据：OmniRouter (arXiv:2502.20576) 约束优化思路 + PID 比例控制。

关于指标来源（与方案文档的差异）
--------------------------------
方案文档 3.2 让控制器每 5s 直接 `await metrics_store.window_stats()`。落地时
发现一个致命细节：**sqlite3 连接不可跨线程**（`check_same_thread=False` 只解
锁断言，不提供并发保护），而 MetricsStore 内部全程 `threading.Lock` + 单连接，
是同步实现；在事件循环里 await 它需要经过 `asyncio.to_thread`，但包装进来的
自定义 store 未必有 `window_stats`。

因此本控制器支持两种指标来源，优先级：
    1. `metrics_provider(since) -> dict`  —— 同步、非阻塞（首选；由
       openai_server 注入基于 **内存计数器** 的实现，零 SQL、零锁竞争）
    2. `metrics_store.window_stats(since)` —— 兼容方案文档的 SQLite 路径

选择内存计数器而非每 5s 查一次 SQLite 的理由：线上 metrics.db 由
MetricsMiddleware 每请求写入 + Dashboard 高频查询，网关侧再叠加 5s 一次的
聚合查询会平白引入写锁竞争；而 τ 只需要「窗口内速率」这一个标量，进程内
累加即可精确得到，还顺带把 BERT RPC 挤出的时间还给了请求路径。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholdConfig:
    """自适应阈值配置（后续可通过 ConfigStore 热更新）。"""

    enabled: bool = True
    # 基础阈值（无预算压力时的默认值）
    tau_base: float = 0.5
    # 阈值上下限（硬质量防线）
    tau_min: float = 0.35
    tau_max: float = 0.75
    # 比例增益（成本超标率 → 阈值上调量）；建议 0.5-2.0
    k_p: float = 1.0
    # 预算：每分钟最大 Token 消耗（0 = 不限）
    budget_tokens_per_min: float = 0.0
    # 延迟 SLA：P90 延迟上限（ms）（0 = 不限）
    latency_sla_ms: float = 0.0
    # 后台采样间隔（秒）
    sample_interval_sec: float = 5.0
    # 滑动窗口大小（秒）
    window_sec: float = 60.0


class AdaptiveThresholdController:
    """自适应阈值控制器。

    职责：
    1. 后台协程定时读取滑动窗口内的成本/延迟指标
    2. 根据成本超标率计算阈值偏移量 Δτ 并钳制到 [τ_min, τ_max]
    3. 提供 get_effective_threshold() / decide() 供 Controller 路由决策调用
    4. 提供 should_backpressure() 判断是否触发背压（429）

    线程/协程安全：`_effective_tau` 等标量用 float/int，CPython 中赋值是原子的，
    且后台协程与请求协程同处一个事件循环，无需加锁。
    """

    def __init__(
        self,
        config: AdaptiveThresholdConfig,
        metrics_store: Optional[Any] = None,
        metrics_provider: Optional[Callable[[float], Dict[str, Any]]] = None,
    ):
        self.config = config
        self.metrics_store = metrics_store
        self.metrics_provider = metrics_provider
        self._effective_tau: float = config.tau_base
        self._cost_rate: float = 0.0   # 窗口内 tokens/min
        self._latency_p90: float = 0.0  # 窗口内 P90 延迟（ms）
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------ 决策

    def get_effective_threshold(self) -> float:
        """当前生效的阈值 τ(t)。Controller 在路由决策时调用它替代静态阈值。"""
        return self._effective_tau

    def decide(self, win_rate: float) -> str:
        """按硬质量防线 + 动态阈值判决档位。

        Returns:
            "strong" / "weak"
        """
        # 硬质量防线①：s ≥ τ_max 一律判强，不受预算压力影响。
        # 这里必须用配置的 τ_max 而不是 _effective_tau —— 后者被钳制在
        # [τ_min, τ_max] 内，用它可以保证「高难请求永不被误降级」。
        if win_rate >= self.config.tau_max:
            return "strong"
        # 硬质量防线②：s < τ_min 是安全降本区，稳定走弱。
        if win_rate < self.config.tau_min:
            return "weak"
        # 中间带：按 τ(t) 浮动分流
        return "strong" if win_rate >= self._effective_tau else "weak"

    def should_backpressure(self, win_rate: float) -> bool:
        """是否应对高难请求触发背压（429）。

        当预算耗尽（cost_rate > budget × 1.5）且 win_rate ≥ τ_max 时返回 True
        —— 不以次充好，宁可让客户端稍后重试，也不静默降级给弱模型。
        """
        if self.config.budget_tokens_per_min <= 0:
            return False
        if win_rate < self.config.tau_max:
            return False
        return self._cost_rate > self.config.budget_tokens_per_min * 1.5

    # ------------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        """启动后台采样协程。"""
        if self._task is not None or not self.config.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._sample_loop())
        logger.info(
            "自适应阈值控制器已启动: tau=[%s, %s] base=%s k_p=%s budget=%s tok/min "
            "latency_sla=%s ms window=%ss",
            self.config.tau_min, self.config.tau_max,
            self.config.tau_base, self.config.k_p,
            self.config.budget_tokens_per_min,
            self.config.latency_sla_ms,
            self.config.window_sec,
        )

    async def stop(self) -> None:
        """停止后台采样协程。"""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _sample_loop(self) -> None:
        """后台采样循环：定时读取指标 → 更新阈值。

        单次失败不应中断循环（保持当前 τ 值），否则一次瞬时抖动会让闭环永久停摆。
        """
        while self._running:
            try:
                await self._update_threshold()
            except Exception as e:  # noqa: BLE001
                logger.warning("自适应阈值采样失败（保持当前值）: %s", e)
            await asyncio.sleep(self.config.sample_interval_sec)

    # ------------------------------------------------------------------ 阈值计算

    async def _update_threshold(self) -> None:
        """读取窗口指标，计算并更新阈值。"""
        stats = await self._fetch_window_stats()
        if stats is None:
            return

        self._cost_rate = float(stats.get("tokens_per_min", 0.0) or 0.0)
        self._latency_p90 = float(stats.get("latency_p90", 0.0) or 0.0)

        delta = self._compute_delta(self._cost_rate, self._latency_p90)
        self._apply_delta(delta)

    async def _fetch_window_stats(self) -> Optional[Dict[str, Any]]:
        """取窗口聚合指标。两种来源都不存在时返回 None（保持当前 τ）。"""
        since = time.time() - self.config.window_sec

        # 首选：同步非阻塞 provider（内存计数器）
        if self.metrics_provider is not None:
            return self.metrics_provider(since)

        if self.metrics_store is None:
            return None

        # 兼容路径：MetricsStore.window_stats（内部走 sqlite，串行执行不阻塞循环）
        fn = getattr(self.metrics_store, "window_stats", None)
        if fn is None:
            logger.warning("metrics_store 无 window_stats，自适应阈值保持静态")
            return None
        result = fn(since)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _compute_delta(self, cost_rate: float, latency_p90: float) -> float:
        """成本超标率 + 延迟超标率 → 阈值偏移 Δτ。"""
        delta = 0.0

        # 成本维度：超预算 → Δτ > 0 → τ 上调 → 更多走弱（省钱）
        if self.config.budget_tokens_per_min > 0:
            cost_gap = cost_rate - self.config.budget_tokens_per_min
            cost_ratio = cost_gap / max(self.config.budget_tokens_per_min, 1.0)
            delta += self.config.k_p * cost_ratio

        # 延迟维度：超 SLA → Δτ > 0 → τ 上调 → 更多走弱（更快）
        # 权重减半：延迟不是成本指标，且受下游波动影响大
        if self.config.latency_sla_ms > 0 and latency_p90 > 0:
            lat_gap = latency_p90 - self.config.latency_sla_ms
            lat_ratio = lat_gap / max(self.config.latency_sla_ms, 1.0)
            delta += self.config.k_p * 0.5 * lat_ratio

        return delta

    def _apply_delta(self, delta: float) -> None:
        new_tau = self.config.tau_base + delta
        new_tau = max(self.config.tau_min, min(self.config.tau_max, new_tau))

        old_tau = self._effective_tau
        self._effective_tau = new_tau

        if abs(new_tau - old_tau) > 0.001:
            logger.info(
                "阈值更新: %.4f → %.4f (cost_rate=%.0f tok/min, latency_p90=%.0f ms, "
                "delta=%.4f)",
                old_tau, new_tau, self._cost_rate, self._latency_p90, delta,
            )

    def _update_threshold_sync_for_test(self) -> None:
        """测试辅助：用已设置的 _cost_rate / _latency_p90 同步重算阈值。"""
        self._apply_delta(self._compute_delta(self._cost_rate, self._latency_p90))

    # ------------------------------------------------------------------ 视图

    def get_status(self) -> Dict[str, Any]:
        """当前状态（供 Dashboard / 验收展示）。"""
        return {
            "enabled": self.config.enabled,
            "effective_tau": round(self._effective_tau, 4),
            "tau_base": self.config.tau_base,
            "tau_min": self.config.tau_min,
            "tau_max": self.config.tau_max,
            "k_p": self.config.k_p,
            "cost_rate": round(self._cost_rate, 1),
            "latency_p90": round(self._latency_p90, 1),
            "budget_tokens_per_min": self.config.budget_tokens_per_min,
            "latency_sla_ms": self.config.latency_sla_ms,
            "window_sec": self.config.window_sec,
            "sample_interval_sec": self.config.sample_interval_sec,
        }


class WindowMetricsCounter:
    """自适应阈值的默认指标来源：进程内滑动窗口计数器。

    为什么不用 MetricsStore SQLite
    ------------------------------
    - 每 5s 一次全表聚合会与 MetricsMiddleware 的写入、Dashboard 的查询
      争抢同一把锁；网关侧不该为「一个标量」付出 SQL 代价。
    - `sqlite3.Connection` 不可跨线程使用，异步化包装容易踩坑。

    语义与 `MetricsStore.window_stats()` 对齐：tokens/min + P90 延迟（近似）。
    P90 用「分桶直方图」近似而非全量排序 —— 只需 1KB 级别内存，且
    τ 是慢变量（5s 采样 + 钳制区间），分桶精度完全够用。
    """

    _LATENCY_BUCKET_MS = 100.0
    _LATENCY_BUCKETS = 600  # 0..60s，超出落最后一桶

    def __init__(self, window_sec: float = 60.0):
        self.window_sec = window_sec
        self._samples: list = []          # [(ts, tokens)] 滚动窗口
        self._lat_hist = [0] * self._LATENCY_BUCKETS

    def record(
        self, tokens: int, latency_ms: float = 0.0, ts: Optional[float] = None
    ) -> None:
        """每请求调用一次。"""
        now = ts if ts is not None else time.time()
        self._samples.append((now, int(tokens)))

        idx = int(latency_ms // self._LATENCY_BUCKET_MS)
        if idx < 0:
            idx = 0
        if idx >= self._LATENCY_BUCKETS:
            idx = self._LATENCY_BUCKETS - 1
        self._lat_hist[idx] += 1

    def record_request_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """兼容调用方按 prompt/completion 分别上报的写法。"""
        self.record(int(prompt_tokens or 0) + int(completion_tokens or 0))

    def _prune(self, since: float) -> None:
        """丢弃窗口外的样本（按时间戳递增，从头部弹，摊还 O(1)）。"""
        cut = 0
        for ts, _ in self._samples:
            if ts >= since:
                break
            cut += 1
        if cut:
            del self._samples[:cut]

    def _reset_hist(self) -> None:
        self._lat_hist = [0] * self._LATENCY_BUCKETS

    def snapshot(self, since: Optional[float] = None) -> Dict[str, Any]:
        """返回与 MetricsStore.window_stats() 同构的统计。"""
        if since is None:
            since = time.time() - self.window_sec
        self._prune(since)

        tokens = sum(t for _, t in self._samples)
        n = len(self._samples)
        elapsed_min = max((time.time() - since) / 60.0, 0.1)

        # P90：桶计数累积到 90% 分位所在桶，取桶上界近似
        total_buckets = sum(self._lat_hist)
        p90 = 0.0
        if total_buckets:
            target = total_buckets * 0.90
            acc = 0
            for i, cnt in enumerate(self._lat_hist):
                acc += cnt
                if acc >= target:
                    p90 = (i + 1) * self._LATENCY_BUCKET_MS
                    break

        return {
            "tokens_per_min": tokens / elapsed_min,
            "latency_p90": round(p90, 2),
            "strong_count": 0,
            "weak_count": 0,
            "total_count": n,
            "total_cost_usd": 0.0,
        }
