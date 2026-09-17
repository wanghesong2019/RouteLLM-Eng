#!/usr/bin/env python
"""对比 compute_elo_mle_with_tie 的求解器性能与结果一致性。

背景：
    实测 LogisticRegression.fit 是路由延迟的 90%（362ms / 110722x10）。
    数据特点：样本极多（110k）、特征极少（10，即模型数）。
    sklearn 默认 lbfgs 不是这种形状的最优选择。

目标：
    找出「结果与 lbfgs 一致、但更快」的求解器/配置。

候选：
    - lbfgs（现状，baseline）
    - newton-cholesky（专为 n_samples >> n_features 设计）
    - liblinear
    - lbfgs + tol 放宽 / max_iter 限制
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_xy(df, sample_weight=None):
    """复刻 compute_elo_mle_with_tie 的 X/Y 构造（不含 fit）。"""
    import math

    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    df2 = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df2.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df2["model_a"]]] = +math.log(10)
    X[np.arange(n), models[df2["model_b"]]] = -math.log(10)

    Y = np.zeros(n)
    Y[df2["winner"] == "model_a"] = 1.0
    tie_idx = (df2["winner"] == "tie") | (df2["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    if sample_weight is not None:
        sample_weight = np.concatenate([sample_weight, sample_weight])
    return X, Y, sample_weight, models.index


def main() -> int:
    from routellm.routers.routers import SWRankingRouter
    from sklearn.linear_model import LogisticRegression

    battles = sys.argv[1]
    embeddings = sys.argv[2]
    model_path = sys.argv[3]

    router = SWRankingRouter(
        local_battles_csv=battles,
        local_embeddings_npy=embeddings,
        local_embedder_path=model_path,
    )
    prompt = "What is the capital of France?"
    emb = router._encode_prompt(prompt)
    sims = np.dot(router.arena_conv_embedding, emb)
    w = router.get_weightings(sims)

    X, Y, sw, model_names = build_xy(router.arena_df, sample_weight=w)
    print(f"X={X.shape} Y={Y.shape} n_models={X.shape[1]}")
    print(f"X 密度: {(X != 0).sum() / X.size * 100:.3f}% (稀疏)\n")

    configs = [
        ("lbfgs (baseline)", dict(fit_intercept=False, penalty=None)),
        ("newton-cholesky", dict(fit_intercept=False, penalty=None, solver="newton-cholesky")),
        ("liblinear", dict(fit_intercept=False, penalty=None, solver="liblinear")),
        ("lbfgs tol=1e-2", dict(fit_intercept=False, penalty=None, tol=1e-2)),
        ("lbfgs max_iter=20", dict(fit_intercept=False, penalty=None, max_iter=20)),
    ]

    base_coef = None
    rows = []
    for name, kwargs in configs:
        times = []
        coef = None
        err = None
        for _ in range(5):
            lr = LogisticRegression(**kwargs)
            t0 = time.perf_counter()
            try:
                lr.fit(X, Y, sample_weight=sw)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                break
            times.append((time.perf_counter() - t0) * 1000)
            coef = lr.coef_[0]

        if err:
            rows.append({"solver": name, "error": err})
            print(f"{name:<22} ERROR: {err}")
            continue

        if base_coef is None:
            base_coef = coef
        max_abs_diff = float(np.max(np.abs(coef - base_coef)))
        # elo 分 = 400 * coef + 1000，转成 elo 差值看实用性
        elo_diff = max_abs_diff * 400
        rows.append(
            {
                "solver": name,
                "mean_ms": round(float(np.mean(times)), 2),
                "min_ms": round(float(np.min(times)), 2),
                "max_abs_coef_diff": round(max_abs_diff, 8),
                "max_elo_diff": round(elo_diff, 4),
            }
        )
        print(
            f"{name:<22} {np.mean(times):>8.1f} ms  "
            f"coef_max_diff={max_abs_diff:.2e}  elo_max_diff={elo_diff:.4f}"
        )

    print("\n" + json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
