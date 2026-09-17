"""验证：Elo 回归对样本权重的敏感度到底有多大。

背景：实测不同 prompt 的权重变化不小（w_mean 全距 17.08），
但 strong/weak 的 elo 差值只变 3.13 分。需要确认：
  1. Elo 回归是否对权重的「整体缩放」不敏感（尺度不变性）
  2. 极端权重（如只保留 top-100 相似样本）能否产生大得多的 elo 差值变化
  3. 若把权重影响放大（如提高 get_weightings 的指数），效果如何
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATTLES = "/mnt/data/wanghesong/routellm/arena_train.csv"
EMB = "/mnt/data/wanghesong/routellm/embeddings/arena_embeddings.npy"
MODEL = "/mnt/data/wanghesong/routellm/models/bge-m3/models/BAAI--bge-m3/snapshots/master"


def main() -> int:
    from routellm.routers.routers import SWRankingRouter
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    r = SWRankingRouter(
        local_battles_csv=BATTLES, local_embeddings_npy=EMB, local_embedder_path=MODEL,
    )
    from routellm.routers.similarity_weighted.local_embedder import LocalBGEM3Embedder

    em = LocalBGEM3Embedder(model_path=MODEL)

    STRONG, WEAK = r.strong_model, r.weak_model
    sidx, widx = r.model2tier[STRONG], r.model2tier[WEAK]

    def elo_diff(weights):
        elo = compute_elo_mle_with_tie(r.arena_df, sample_weight=weights)
        return float(elo[sidx] - elo[widx])

    def wr(diff):
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    prompts = ["hi", "Prove that the square root of 2 is irrational using proof by contradiction."]

    print("=" * 88)
    print("对照 1：权重整体缩放是否影响 elo 差值（尺度不变性）")
    print("=" * 88)
    v = em.encode_prompts([prompts[0]])[0]
    base_w = r.get_weightings(r.arena_conv_embedding @ v)
    print(f"{'缩放系数':>12}{'elo_diff':>12}{'winrate':>12}")
    print("-" * 88)
    for k in [0.01, 0.1, 1.0, 10.0, 100.0]:
        d = elo_diff(base_w * k)
        print(f"{k:>12}{d:>12.6f}{wr(d):>12.6f}")

    print("\n" + "=" * 88)
    print("对照 2：极端权重（只保留 top-K 最相似样本）")
    print("=" * 88)
    print(f"{'topK':>8}{'elo_diff':>12}{'winrate':>12}")
    print("-" * 88)
    sims = r.arena_conv_embedding @ v
    order = np.argsort(sims)[::-1]
    for K in [10, 100, 1000, 10000, 55361]:
        w = np.zeros(len(sims))
        w[order[:K]] = 1.0
        d = elo_diff(w)
        print(f"{K:>8}{d:>12.6f}{wr(d):>12.6f}")

    print("\n" + "=" * 88)
    print("对照 3：不同 prompt 在不同 top-K 下的 elo 差值（是否有区分度）")
    print("=" * 88)
    for K in [100, 1000, 10000, 55361]:
        diffs = []
        for p in prompts:
            vv = em.encode_prompts([p])[0]
            ss = r.arena_conv_embedding @ vv
            o = np.argsort(ss)[::-1]
            w = np.zeros(len(ss))
            w[o[:K]] = 1.0
            diffs.append(elo_diff(w))
        print(f"  topK={K:>6}  diffs={[round(d,3) for d in diffs]}  全距={max(diffs)-min(diffs):.4f}  "
              f"winrate全距={wr(max(diffs))-wr(min(diffs)):.6f}")

    print("\n" + "=" * 88)
    print("对照 4：把 Elo 回归的数值精度放大（SCALE 参数）")
    print("=" * 88)
    print("  注：winrate 用 400 为尺度，若 elo 分整体放大，差值也等比放大")
    print(f"{'SCALE':>8}{'elo_diff':>12}{'winrate(用400尺度)':>22}")
    print("-" * 88)
    for scale in [400, 4000, 40000]:
        elo = compute_elo_mle_with_tie(r.arena_df, sample_weight=base_w, SCALE=scale)
        d = float(elo[sidx] - elo[widx])
        print(f"{scale:>8}{d:>12.3f}{wr(d):>22.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
