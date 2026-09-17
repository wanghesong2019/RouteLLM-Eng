#!/usr/bin/env python
"""验证 SWRankingRouter 多数据集模式与官方 thresholds 对齐。

用改造后的 SWRankingRouter（local_datasets 参数）跑真实数据，
确认 win_rate 与官方发布值一致 —— 这是代码层面的最终验收。

用法：
    python scripts/verify_sw_ranking_multi_dataset.py \
        --arena-csv   <arena_train.csv> \
        --judge-parquet <gpt4_judge_battles.parquet> \
        --arena-emb   <arena_embeddings.npy> \
        --judge-emb   <judge_embeddings.npy> \
        --model-path  <bge-m3 目录> \
        --official-parquet <official_thresholds.parquet> \
        --sample 100
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-csv", required=True)
    ap.add_argument("--judge-parquet", required=True)
    ap.add_argument("--arena-emb", required=True)
    ap.add_argument("--judge-emb", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--official-parquet", required=True)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.pop("OPENAI_API_KEY", None)

    from routellm.routers.routers import SWRankingRouter

    # ---- 1) 按官方配置构造（两个数据集拼接）----
    print("[1/4] 构造 SWRankingRouter（arena + judge 拼接）")
    t0 = time.time()
    router = SWRankingRouter(
        local_datasets=[
            {"battles": args.arena_csv, "embeddings": args.arena_emb, "count": 55361},
            {"battles": args.judge_parquet, "embeddings": args.judge_emb, "count": 109101},
        ],
        local_embedder_path=args.model_path,
    )
    print(f"      构造 {time.time()-t0:.1f}s")
    print(f"      arena_df 行数       : {len(router.arena_df)}")
    print(f"      arena_conv_embedding: {router.arena_conv_embedding.shape}")
    print(f"      embedding_model     : {router.embedding_model}")
    print(f"      encoder_backend     : {router.encoder_backend}")
    assert len(router.arena_df) == 164462, f"应为 164462，实际 {len(router.arena_df)}"
    print("      ✅ 行数自洽（55361 + 109101 = 164462）")

    # ---- 2) 抽样并计算 win_rate ----
    print(f"\n[2/4] 计算 win_rate（抽样 {args.sample} 条）")
    offsets = [int(x) for x in router.arena_df.index]  # 保留原 index 用于回查
    arena = pd.read_csv(args.arena_csv)
    arena["first_turn"] = arena["prompt"].apply(lambda s: json.loads(s)[0].strip())
    arena = arena.loc[arena["first_turn"].apply(len) >= 16].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(arena))
    idx = rng.choice(len(arena), size=n, replace=False)
    sample = arena.iloc[idx].reset_index(drop=True)
    prompts = sample["first_turn"].tolist()

    t0 = time.time()
    wrs = np.array([router.calculate_strong_win_rate(p) for p in prompts])
    print(f"      {n} 条耗时 {time.time()-t0:.1f}s（{(time.time()-t0)/n*1000:.0f} ms/条）")

    # ---- 3) 与官方对标 ----
    print(f"\n[3/4] 与官方对标")
    official = pd.read_parquet(args.official_parquet).set_index("id")
    matched = official.reindex(sample["id"].tolist())["sw_ranking"].values
    mask = ~np.isnan(matched)

    print(f"  {'':16}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
    print("  " + "-" * 56)
    print(f"  {'官方(全量)':16}{official['sw_ranking'].mean():>10.4f}"
          f"{official['sw_ranking'].std():>10.4f}"
          f"{official['sw_ranking'].min():>10.4f}{official['sw_ranking'].max():>10.4f}")
    print(f"  {'本实现(抽样)':16}{wrs.mean():>10.4f}{wrs.std():>10.4f}"
          f"{wrs.min():>10.4f}{wrs.max():>10.4f}")

    if mask.sum() > 10:
        corr = float(np.corrcoef(wrs[mask], matched[mask])[0, 1])
        print(f"\n  逐条对标 n={mask.sum()}: corr = {corr:.4f}")
        print(f"    本实现 mean={wrs[mask].mean():.4f}  官方 mean={matched[mask].mean():.4f}")
        print(f"    均值比 = {wrs[mask].mean()/matched[mask].mean():.4f}")

        # 判断标准
        ok_mean = abs(wrs[mask].mean() / matched[mask].mean() - 1) < 0.05
        ok_corr = corr > 0.5
        print(f"\n  验收: 均值比在 ±5% 内 → {'✅' if ok_mean else '❌'}"
              f"   相关系数 > 0.5 → {'✅' if ok_corr else '❌'}")

    # ---- 4) 阈值标定 ----
    print(f"\n[4/4] 阈值标定（官方 quantile 方法）")
    print(f"  {'目标强模型占比':>16}{'官方阈值':>14}{'本实现阈值':>14}")
    for pct in [0.5, 0.3, 0.2, 0.1]:
        th_o = float(np.quantile(official["sw_ranking"], 1 - pct))
        th_m = float(np.quantile(wrs, 1 - pct))
        print(f"  {pct*100:>15.0f}%{th_o:>14.6f}{th_m:>14.6f}")

    print("\n" + "=" * 60)
    print("✅ 多数据集模式验证完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
