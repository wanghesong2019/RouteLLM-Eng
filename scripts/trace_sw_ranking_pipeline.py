"""定位 sw_ranking 区分度不足的根因。

链路：prompt → bge-m3 向量 → 与 55k 向量算 cosine → get_weightings
      → 加权 Elo 回归 → strong_winrate

对一批 prompt 逐环节测量中间量，找出"哪一环丢失了区分度"：
  环节 A: bge-m3 向量本身（prompt 之间是否有差异）
  环节 B: cosine 相似度分布（形状是否有差异）
  环节 C: weightings（归一化后是否趋同）
  环节 D: 加权 Elo 回归结果（elo 分是否有差异）
  环节 E: 最终 winrate
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
        local_battles_csv=BATTLES,
        local_embeddings_npy=EMB,
        local_embedder_path=MODEL,
    )

    # 挑选语义差异极大的 prompt
    prompts = [
        "hi",
        "What is 1+1?",
        "What is the capital of France?",
        "Write a haiku about autumn leaves.",
        "Explain the trade-offs between consistency and availability in distributed systems.",
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "Derive the Navier-Stokes equations from first principles for an incompressible Newtonian fluid.",
    ]

    print("=" * 90)
    print("环节 A：bge-m3 向量本身（prompt 间差异）")
    print("=" * 90)
    vecs = r._encode_prompt.__self__ and None  # placeholder
    from routellm.routers.similarity_weighted.local_embedder import LocalBGEM3Embedder

    emb_model = LocalBGEM3Embedder(model_path=MODEL)
    V = emb_model.encode_prompts(prompts)
    print(f"  向量矩阵 {V.shape}, dtype={V.dtype}")
    # prompt 之间的余弦相似度（已归一化→点积）
    sim_mat = V @ V.T
    off = sim_mat[np.triu_indices(len(prompts), k=1)]
    print(f"  prompt 间相似度: min={off.min():.4f} max={off.max():.4f} mean={off.mean():.4f} std={off.std():.4f}")
    print(f"  → 说明 bge-m3 能区分这些 prompt（相似度 0.3~0.7 区间）")

    print("\n" + "=" * 90)
    print("环节 B/C/D/E：逐环中间量")
    print("=" * 90)
    print(f"{'prompt':<50}{'sim_max':>9}{'sim_mean':>10}{'w_min':>9}{'w_mean':>10}")
    print("-" * 90)

    rows = []
    for p in prompts:
        v = emb_model.encode_prompts([p])[0]
        sims = r.arena_conv_embedding @ v
        w = r.get_weightings(sims)
        elo = compute_elo_mle_with_tie(r.arena_df, sample_weight=w)
        ws = elo[r.model2tier[r.weak_model]]
        ss = elo[r.model2tier[r.strong_model]]
        wr = 1 - 1 / (1 + 10 ** ((ss - ws) / 400))
        rows.append({
            "prompt": p[:48],
            "sim_max": sims.max(),
            "sim_mean": sims.mean(),
            "w_min": w.min(),
            "w_mean": w.mean(),
            "elo_strong": ss,
            "elo_weak": ws,
            "winrate": wr,
        })
        print(f"{p[:48]:<50}{sims.max():>9.4f}{sims.mean():>10.4f}{w.min():>9.3f}{w.mean():>10.3f}")

    print("\n" + "=" * 90)
    print("环节 D/E：Elo 分与最终 winrate")
    print("=" * 90)
    print(f"{'prompt':<50}{'elo_strong':>12}{'elo_weak':>11}{'winrate':>11}")
    print("-" * 90)
    for row in rows:
        print(f"{row['prompt'][:48]:<50}{row['elo_strong']:>12.3f}{row['elo_weak']:>11.3f}{row['winrate']:>11.6f}")

    # 各环节的"变异系数"对比
    print("\n" + "=" * 90)
    print("各环节变异程度（衡量区分度传递）")
    print("=" * 90)

    def spread(name, vals):
        vals = np.array(vals)
        print(f"  {name:<28} 全距={vals.max()-vals.min():>12.6f}  std={vals.std():>12.6f}")

    spread("C: w_min", [x["w_min"] for x in rows])
    spread("C: w_mean", [x["w_mean"] for x in rows])
    spread("B: sim_max", [x["sim_max"] for x in rows])
    spread("B: sim_mean", [x["sim_mean"] for x in rows])
    spread("D: elo_strong", [x["elo_strong"] for x in rows])
    spread("D: elo_weak", [x["elo_weak"] for x in rows])
    spread("E: winrate", [x["winrate"] for x in rows])

    print("\n【关键】elo_strong 与 elo_weak 的差值（决定 winrate）")
    diffs = [x["elo_strong"] - x["elo_weak"] for x in rows]
    print(f"  elo 差值: min={min(diffs):.4f} max={max(diffs):.4f} 全距={max(diffs)-min(diffs):.4f}")
    print(f"  → winrate = 1/(1+10^(-diff/400))，diff 变化这么小 → winrate 必然挤在一起")

    print("\n" + "=" * 90)
    print("猜想的直接验证：elo_strong 与 elo_weak 是否'同增同减'")
    print("=" * 90)
    es = np.array([x["elo_strong"] for x in rows])
    ew = np.array([x["elo_weak"] for x in rows])
    print(f"  corr(elo_strong, elo_weak) = {np.corrcoef(es, ew)[0,1]:.6f}")
    print(f"  → 若接近 1.0，说明两者同步移动，差值被抵消（推测根因）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
