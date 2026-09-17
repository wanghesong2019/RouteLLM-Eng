"""Prometheus 指标端点。

设计依据：方案文档 4.2.3 —— 「同时暴露 Prometheus metrics 端点，兼容标准监控生态」。

为什么不用 prometheus_client：
    网关镜像刻意保持轻量（见 requirements-gateway.txt 与 ADR-001）。
    Prometheus 的文本暴露格式非常简单（几行字符串拼接），
    自己实现可避免引入额外依赖，且完全可控。

支持的指标类型：
    counter   单调递增计数（如请求总数）
    gauge     可增可减的瞬时值（如当前缓存条目数）
    histogram 观测值分布（如延迟，输出 _bucket/_sum/_count）

线程安全：所有写操作加锁（网关是多线程环境）。
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

_lock = threading.Lock()

# metric_name -> {label_str -> value}
_counters: Dict[str, Dict[str, float]] = {}
_gauges: Dict[str, Dict[str, float]] = {}

# metric_name -> {"buckets": [le...], "counts": [n...], "sum": float, "count": int}
# 按 label 分组的直方图另存
_histograms: Dict[str, Dict[str, Dict]] = {}

# 默认直方图桶（毫秒），覆盖 1ms ~ 10s
DEFAULT_BUCKETS: List[float] = [
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
]


def _label_str(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _split_labels(label_str: str) -> str:
    return label_str or ""


def reset() -> None:
    """清空所有指标（测试用）。"""
    with _lock:
        _counters.clear()
        _gauges.clear()
        _histograms.clear()


# ------------------------------------------------------------------ counter


def inc_counter(
    name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0
) -> None:
    ls = _label_str(labels or {})
    with _lock:
        _counters.setdefault(name, {})
        _counters[name][ls] = _counters[name].get(ls, 0.0) + amount


def get_counter(name: str, labels: Optional[Dict[str, str]] = None) -> float:
    ls = _label_str(labels or {})
    with _lock:
        return _counters.get(name, {}).get(ls, 0.0)


# ------------------------------------------------------------------ gauge


def set_gauge(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
) -> None:
    ls = _label_str(labels or {})
    with _lock:
        _gauges.setdefault(name, {})[ls] = float(value)


def get_gauge(name: str, labels: Optional[Dict[str, str]] = None) -> float:
    ls = _label_str(labels or {})
    with _lock:
        return _gauges.get(name, {}).get(ls, 0.0)


# ------------------------------------------------------------------ histogram


def observe_histogram(
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None,
    buckets: Optional[List[float]] = None,
) -> None:
    buckets = list(buckets) if buckets else list(DEFAULT_BUCKETS)
    ls = _label_str(labels or {})
    with _lock:
        by_label = _histograms.setdefault(name, {})
        h = by_label.setdefault(
            ls,
            {"buckets": list(buckets), "counts": [0] * len(buckets), "sum": 0.0, "count": 0},
        )
        for i, le in enumerate(h["buckets"]):
            if value <= le:
                h["counts"][i] += 1
        h["sum"] += float(value)
        h["count"] += 1


# ------------------------------------------------------------------ render


def render_metrics() -> str:
    """渲染为 Prometheus 文本暴露格式。"""
    lines: List[str] = []

    with _lock:
        for name, series in sorted(_counters.items()):
            lines.append(f"# TYPE {name} counter")
            for ls, v in sorted(series.items()):
                lines.append(f"{name}{ls} {_fmt(v)}")

        for name, series in sorted(_gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            for ls, v in sorted(series.items()):
                lines.append(f"{name}{ls} {_fmt(v)}")

        for name, by_label in sorted(_histograms.items()):
            lines.append(f"# TYPE {name} histogram")
            for ls, h in sorted(by_label.items()):
                # 把 label 合进 le 标签
                for le, cnt in zip(h["buckets"], h["counts"]):
                    if ls:
                        merged = ls[:-1] + f',le="{le}"' + "}"
                    else:
                        merged = f'{{le="{le}"}}'
                    lines.append(f"{name}_bucket{merged} {cnt}")
                # +Inf 桶
                if ls:
                    merged = ls[:-1] + ',le="+Inf"' + "}"
                else:
                    merged = '{le="+Inf"}'
                lines.append(f"{name}_bucket{merged} {h['count']}")

                lines.append(f"{name}_sum{ls} {_fmt(h['sum'])}")
                lines.append(f"{name}_count{ls} {h['count']}")

    return "\n".join(lines) + "\n"


def _fmt(v: float) -> str:
    """Prometheus 数值格式：整数不带小数点。"""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(float(v))
