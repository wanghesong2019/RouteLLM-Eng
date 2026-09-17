"""验证修复方向：拉大 get_weightings 的动态范围。

根因（已确认）：
    get_weightings(sims) = 10 * 10^(sim/max_sim)
    权重区间仅 12~100（动态范围 8 倍），
    → 5.5 万条样本近似等权参与回归 → 不同 prompt 结果趋同（elo_diff 全距 0）

对照实验已证明：
    topK=100（仅最相似 100 条，等效极窄权重）→ elo_diff 全距 87 分，winrate 全距 0.12
    topK=全部（近似等权）→ elo_diff 全距 0，winrate 全距 0

本脚本测试：把 get_weightings 的指数放大 / 改用其他映射，看能否恢复区分度。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATTLES = "/mnt/data/wanghesong/routellm/arena_train.csv"
EMB = "/mnt/data/wanghesong/routellm/embeddings/arena_embeddings.npy"
MODEL = "/mnt/data/wanghesong/routellm/models/bge-m3/models/BAAI--bge-m3/snapshots/master"

PROMPTS = [
    "hi",
    "What is 1+1?",
    "Write a haiku about autumn leaves.",
    "Explain the trade-offs between consistency and availability in distributed systems.",
    "Prove that the square root of 2 is irrational using proof by contradiction.",
]


def main() -> int:
    from routellm.routers.routers import SWRankingRouter
    from routellm.routers.similarity_weighted.local_embedder import LocalBGEM3Embedder
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    r = SWRankingRouter(
        local_battles_csv=BATTLES, local_embeddings_npy=EMB, local_embedder_path=MODEL,
    )
    em = LocalBGEM3Embedder(model_path=MODEL)
    sidx, widx = r.model2tier[r.strong_model], r.model2tier[r.weak_model]

    def elo_diff(w):
        elo = compute_elo_mle_with_tie(r.arena_df, sample_weight=w)
        return float(elo[sidx] - elo[widx])

    def wr(d):
        return 1.0 / (1.0 + 10 ** (-d / 400))

    # ---- 候选权重映射 ----
    def w_orig(sims):
        """上游：10 * 10^(sim/max_sim)，区间 [12, 100]"""
        return 10 * 10 ** (sims / np.max(sims))

    def w_pow(sims, p=4.0):
        """把 [0,1] 归一化后取幂，拉大动态范围"""
        s = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
        return np.exp(p * s)  # 区间 [1, e^p]

    def w_exp(sims, p=10.0):
        """softmax 风格的指数放大"""
        s = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
        return np.exp(p * (s - 1.0))  # 区间 [e^-p, 1]

    def w_rank(sims, keep=0.05):
        """只保留最相似的 keep 比例"""
        k = max(1, int(len(sims) * keep))
        order = np.argsort(sims)[::-1]
        w = np.zeros(len(sims))
        w[order[:k]] = 1.0
        return w

    candidates = {
        "上游 w_orig (12~100)": w_orig,
        "pow p=4  (1~54.6)": lambda s: w_pow(s, 4),
        "pow p=8  (1~2981)": lambda s: w_pow(s, 8),
        "pow p=16 (1~8.9e6)": lambda s: w_pow(s, 16),
        "exp p=10 (4.5e-5~1)": lambda s: w_exp(s, 10),
        "top 10%": lambda s: w_rank(s, 0.10),
        "top 5%": lambda s: w_rank(s, 0.05),
        "top 1%": lambda s: w_rank(s, 0.01),
    }

    print("=" * 96)
    print(f"不同权重映射下的区分度（{len(PROMPTS)} 个语义差异大的 prompt）")
    print("=" * 96)
    print(f"{'映射':<24}{'w区间':>22}{'elo_diff 全距':>16}{'winrate全距':>14}{'评价':>8}")
    print("-" * 96)

    for name, fn in candidates.items():
        diffs = []
        wrange = None
        for p in PROMPTS:
            v = em.encode_prompts([p])[0]
            sims = r.arena_conv_embedding @ v
            w = fn(sims)
            wrange = (float(w.min()), float(w.max()))
            diffs.append(elo_diff(w))
        spread = max(diffs) - min(diffs)
        wr_spread = wr(max(diffs)) - wr(min(diffs))
        verdict = "✅ 好" if wr_spread > 0.05 else ("⚠️ 弱" if wr_spread > 0.01 else "❌ 无")
        print(
            f"{name:<24}{f'{wrange[0]:.3g}~{wrange[1]:.3g}':>22}"
            f"{spread:>16.4f}{wr_spread:>14.6f}{verdict:>8}"
        )

    print("\n" + "=" * 96)
    print("详细：上游映射 vs 建议映射的逐 prompt 结果")
    print("=" * 96)
    for name, fn in [("上游 w_orig", w_orig), ("pow p=8", lambda s: w_pow(s, 8))]:
        print(f"\n--- {name} ---")
        print(f"{'prompt':<52}{'elo_diff':>10}{'winrate':>11}")
        for p in PROMPTS:
            v = em.encode_prompts([p])[0]
            d = elo_diff(fn(r.arena_conv_embedding @ v))
            print(f"{p[:50]:<52}{d:>10.3f}{wr(d):>11.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
