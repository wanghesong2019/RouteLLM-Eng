#!/usr/bin/env python
"""诊断 sw_ranking 的 win_rate 分布，并标定合适的 threshold。

背景：
    sw_ranking 上线后发现 win_rate 全挤在 0.689~0.693，用固定 0.5 阈值
    会导致所有请求都走强模型。查上游 calibrate_threshold.py 发现官方做法是：

        threshold = win_rate 分布的 quantile(q=1 - strong_model_pct)

    即按「期望强模型调用占比」反推阈值，而非固定 0.5。

本脚本：
    1. 在真实 arena 数据上采样 N 条 prompt，计算 sw_ranking 的 win_rate
    2. 输出分布统计（min/p25/p50/p75/p90/p95/max）与直方图
    3. 按几种目标强模型占比，反推对应 threshold
    4. 对照 bert 路由做同样分析（bert 已知有区分度，作为参照系）

用法：
    python scripts/diagnose_sw_ranking_distribution.py \
        --sample-size 500 \
        --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def describe(name: str, vals: np.ndarray) -> dict:
    qs = [0, 5, 25, 50, 75, 90, 95, 100]
    pcts = np.percentile(vals, qs)
    print(f"\n{name}  (n={len(vals)})")
    print("  " + "  ".join(f"p{q}={v:.4f}" for q, v in zip(qs, pcts)))
    print(f"  mean={vals.mean():.4f}  std={vals.std():.4f}  range={vals.max()-vals.min():.4f}")
    return {
        "n": int(len(vals)),
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std()), 4),
        "percentiles": {f"p{q}": round(float(v), 4) for q, v in zip(qs, pcts)},
    }


def calibrate(name: str, vals: np.ndarray, targets=(0.5, 0.3, 0.2, 0.1)) -> dict:
    """按官方方法反推阈值：threshold = quantile(1 - strong_pct)。"""
    out = {}
    print(f"\n{name} 的阈值标定（官方方法：quantile(1-strong_pct)）")
    for pct in targets:
        th = float(np.quantile(vals, 1 - pct))
        actual = float((vals >= th).mean())
        out[f"strong_pct_{int(pct*100)}"] = {
            "threshold": round(th, 5),
            "actual_strong_ratio": round(actual, 4),
        }
        print(f"  目标强模型占比 {pct*100:>5.0f}%  →  threshold={th:.5f}（实测占比 {actual*100:.1f}%）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles-csv", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bert-url", default="http://127.0.0.1:6070",
                    help="BERT 服务地址（对照用；设为空跳过）")
    ap.add_argument("--sample-size", type=int, default=500,
                    help="从 arena 中抽样多少条 prompt 做分布分析")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.pop("OPENAI_API_KEY", None)

    # ---- 1) 准备样本 ----
    print(f"[1/4] 读取 arena: {args.battles_csv}")
    raw = pd.read_csv(args.battles_csv)
    # 复刻官方做法：取每条对战的首轮 prompt
    raw["first_turn"] = raw["prompt"].apply(lambda s: json.loads(s)[0].strip())
    raw = raw.loc[raw["first_turn"].apply(len) >= 16]
    print(f"      preprocess 后 {len(raw)} 条")

    rng = np.random.default_rng(args.seed)
    n = min(args.sample_size, len(raw))
    idx = rng.choice(len(raw), size=n, replace=False)
    prompts = raw.iloc[idx]["first_turn"].tolist()
    print(f"      抽样 {n} 条 prompt 用于分布分析")

    results = {}

    # ---- 2) sw_ranking（本地） ----
    print(f"\n[2/4] 计算 sw_ranking win_rate（本地 bge-m3）")
    from routellm.routers.routers import SWRankingRouter

    t0 = time.perf_counter()
    router = SWRankingRouter(
        local_battles_csv=args.battles_csv,
        local_embeddings_npy=args.embeddings,
        local_embedder_path=args.model_path,
    )
    print(f"      router 构造 {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    sw = np.array([router.calculate_strong_win_rate(p) for p in prompts])
    print(f"      {n} 条耗时 {time.perf_counter()-t0:.1f}s（{(time.perf_counter()-t0)/n*1000:.0f} ms/条）")

    results["sw_ranking"] = describe("sw_ranking (本地 bge-m3)", sw)
    results["sw_ranking"]["calibration"] = calibrate("sw_ranking", sw)

    # ---- 3) bert（对照，走 host 服务） ----
    if args.bert_url:
        print(f"\n[3/4] 计算 bert win_rate（对照，{args.bert_url}）")
        try:
            import urllib.request

            payload = json.dumps({"prompts": prompts}).encode()
            req = urllib.request.Request(
                f"{args.bert_url}/v1/score", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                bd = json.loads(resp.read().decode())
            bt = np.array([r["win_rate"] for r in bd["results"]])
            results["bert"] = describe("bert (对照)", bt)
            results["bert"]["calibration"] = calibrate("bert", bt)

            # 相关性：两个路由的 win_rate 是否相关
            corr = float(np.corrcoef(sw, bt)[0, 1])
            print(f"\n[相关] sw_ranking vs bert 的 win_rate 相关系数: {corr:.4f}")
            results["corr_sw_vs_bert"] = round(corr, 4)
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ bert 服务不可达，跳过对照: {type(e).__name__}: {e}")
    else:
        print("\n[3/4] 跳过 bert 对照")

    # ---- 4) 官方发布的分位数数据集对照 ----
    print(f"\n[4/4] 官方阈值数据集（routellm/lmsys-arena-human-preference-55k-thresholds）")
    print("      注：该数据集含官方各路由的 win_rate，可用于直接对标。")
    print("      本机网络受限（huggingface.co 仅 IPv6 不可达），如需对标请手动获取。")

    print("\n" + "=" * 70)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
