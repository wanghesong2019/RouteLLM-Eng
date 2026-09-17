#!/usr/bin/env python
"""拆解 SWRankingRouter.calculate_strong_win_rate 的耗时构成。

目的：
    实测路由延迟 P50≈509ms。需要精确定位瓶颈：
      - _encode_prompt（本地 bge-m3 推理）
      - 相似度点积（55361×1024）
      - compute_elo_mle_with_tie（LogisticRegression 拟合）
      - 其他（get_weightings / tiers 查表）

用法：
    python scripts/profile_sw_ranking_latency.py \
        --battles-csv <arena_train.csv> \
        --embeddings  <arena_embeddings.npy> \
        --model-path  <bge-m3 权重目录> \
        --runs 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000  # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles-csv", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    os.environ.pop("OPENAI_API_KEY", None)

    from routellm.routers.routers import SWRankingRouter
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    router = SWRankingRouter(
        local_battles_csv=args.battles_csv,
        local_embeddings_npy=args.embeddings,
        local_embedder_path=args.model_path,
    )
    # 预热，排除模型加载
    router.calculate_strong_win_rate("warmup prompt for profiling purposes.")

    prompt = "What is the capital of France, and why is it significant?"
    enc_ms, sim_ms, elo_ms, wgt_ms, total_ms = [], [], [], [], []

    for _ in range(args.runs):
        t_all = time.perf_counter()

        emb, t_enc = timed(router._encode_prompt, prompt)
        enc_ms.append(t_enc)

        _, t_sim = timed(
            lambda: np.dot(router.arena_conv_embedding, emb)
        )
        sim_ms.append(t_sim)

        sims = np.dot(router.arena_conv_embedding, emb)
        w, t_w = timed(router.get_weightings, sims)
        wgt_ms.append(t_w)

        _, t_elo = timed(
            compute_elo_mle_with_tie, router.arena_df, sample_weight=w
        )
        elo_ms.append(t_elo)

        total_ms.append((time.perf_counter() - t_all) * 1000)

    def stat(xs):
        return {
            "mean": round(statistics.mean(xs), 2),
            "p50": round(statistics.median(xs), 2),
            "min": round(min(xs), 2),
            "max": round(max(xs), 2),
        }

    parts = {
        "encode_prompt (bge-m3)": stat(enc_ms),
        "similarity dot (55k x 1024)": stat(sim_ms),
        "get_weightings": stat(wgt_ms),
        "compute_elo_mle_with_tie": stat(elo_ms),
        "TOTAL (sum of parts)": stat(total_ms),
    }

    print("=" * 66)
    print(f"路由延迟拆解（{args.runs} 次，已预热）")
    print("=" * 66)
    print(f"{'阶段':<32}{'mean':>10}{'p50':>10}{'min':>10}{'max':>10}")
    print("-" * 66)
    for name, s in parts.items():
        print(f"{name:<32}{s['mean']:>10.2f}{s['p50']:>10.2f}{s['min']:>10.2f}{s['max']:>10.2f}")
    print("-" * 66)

    tot = statistics.mean(total_ms)
    print("\n占比：")
    for name in list(parts)[:-1]:
        print(f"  {name:<34} {parts[name]['mean']/tot*100:>6.1f}%")
    print(f"  {'（合计）':<34} {sum(parts[k]['mean'] for k in list(parts)[:-1])/tot*100:>6.1f}%")

    # Elo 回归内部：拟合本身 vs 数据准备
    print("\nElo 回归内部拆解：")
    df = router.arena_df
    import math as _math
    import pandas as pd

    t0 = time.perf_counter()
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    df2 = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df2.shape[0]
    X = np.zeros([n, p])
    X[np.arange(n), models[df2["model_a"]]] = +_math.log(10)
    X[np.arange(n), models[df2["model_b"]]] = -_math.log(10)
    Y = np.zeros(n)
    Y[df2["winner"] == "model_a"] = 1.0
    tie_idx = (df2["winner"] == "tie") | (df2["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0
    t_prep = (time.perf_counter() - t0) * 1000

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(fit_intercept=False, penalty=None)
    w = router.get_weightings(np.dot(router.arena_conv_embedding, router._encode_prompt(prompt)))
    sw = np.concatenate([w, w])
    t0 = time.perf_counter()
    lr.fit(X, Y, sample_weight=sw)
    t_fit = (time.perf_counter() - t0) * 1000

    print(f"  X/Y 矩阵构造                 {t_prep:>8.2f} ms")
    print(f"  LogisticRegression.fit       {t_fit:>8.2f} ms")
    print(f"  → 矩阵形状 X={X.shape}, Y={Y.shape}, 模型数 p={p}")

    out = {
        "runs": args.runs,
        "breakdown_ms": parts,
        "elo_internal_ms": {"matrix_prep": round(t_prep, 2), "fit": round(t_fit, 2)},
        "matrix_shape": {"X": list(X.shape), "n_models": int(p)},
    }
    print("\n" + json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
