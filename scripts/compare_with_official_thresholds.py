#!/usr/bin/env python
"""补齐 judge 数据集后，与官方 thresholds 对标验证。

假设：此前 sw_ranking 与官方有 3.2 倍系统偏差（mean 0.692 vs 0.216），
      原因是缺少 gpt4_judge_battles 数据集（官方拼接了 arena + judge）。

本脚本：
    1. 用 arena + judge 拼接后的完整数据构造 SWRankingRouter
    2. 在官方 thresholds 数据集覆盖的同一批 prompt 上计算 win_rate
    3. 与官方 sw_ranking 值逐条对比（均值/标准差/相关系数）
    4. 若接近，则假设成立；否则需另找原因

用法：
    python scripts/compare_with_official_thresholds.py \
        --arena-csv <arena_train.csv> \
        --judge-parquet <gpt4_judge_battles.parquet> \
        --arena-emb <arena_embeddings.npy> \
        --judge-emb <judge_embeddings.npy> \
        --model-path <bge-m3 目录> \
        --official-parquet <official_thresholds.parquet> \
        --sample 500
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
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.pop("OPENAI_API_KEY", None)

    from routellm.routers.similarity_weighted.local_embedder import LocalBGEM3Embedder
    from routellm.routers.similarity_weighted.utils import (
        compute_elo_mle_with_tie,
        compute_tiers,
        preprocess_battles,
    )

    # ---- 1) 官方数据 ----
    print(f"[1/6] 读取官方 thresholds: {args.official_parquet}")
    official = pd.read_parquet(args.official_parquet)
    print(f"      {len(official)} 行, 列={list(official.columns)}")

    # ---- 2) 拼接 arena + judge ----
    print(f"\n[2/6] 拼接数据")
    arena = pd.read_csv(args.arena_csv)
    judge = pd.read_parquet(args.judge_parquet)
    print(f"      arena {len(arena)} 条 + judge {len(judge)} 条")

    combined = pd.concat([arena, judge], ignore_index=True)
    arena_df = preprocess_battles(combined.copy())
    print(f"      preprocess 后: {len(arena_df)} 条")

    emb = np.concatenate([
        np.load(args.arena_emb).astype(np.float32),
        np.load(args.judge_emb).astype(np.float32),
    ])
    print(f"      向量拼接: {emb.shape}")
    assert len(arena_df) == len(emb), (
        f"行数 {len(arena_df)} 与向量 {len(emb)} 不一致"
    )
    print(f"      ✅ 行数自洽")

    # ---- 3) 构造分档 ----
    print(f"\n[3/6] 计算 Elo 分档（{len(arena_df)} 条）")
    t0 = time.time()
    model_ratings = compute_elo_mle_with_tie(arena_df)
    model2tier = compute_tiers(model_ratings, num_tiers=10)
    print(f"      {time.time()-t0:.1f}s，模型数 {len(model_ratings)}")
    print(f"      Elo top5: {[(k, round(v,1)) for k,v in list(model_ratings.items())[:5]]}")

    tiered = arena_df.copy()
    tiered["model_a"] = tiered["model_a"].apply(lambda x: model2tier[x])
    tiered["model_b"] = tiered["model_b"].apply(lambda x: model2tier[x])

    STRONG = "gpt-4-1106-preview"
    WEAK = "mixtral-8x7b-instruct-v0.1"
    print(f"      strong={STRONG} elo={model_ratings.get(STRONG)}")
    print(f"      weak  ={WEAK} elo={model_ratings.get(WEAK)}")
    if STRONG in model_ratings and WEAK in model_ratings:
        print(f"      elo 差值 = {model_ratings[STRONG]-model_ratings[WEAK]:.3f}")

    # ---- 4) 抽样 prompt 并计算 win_rate ----
    print(f"\n[4/6] 计算 win_rate（抽样 {args.sample} 条）")
    raw = combined.copy()
    import json as _json
    raw["first_turn"] = raw["prompt"].apply(lambda s: _json.loads(s)[0].strip())
    raw = raw.loc[raw["first_turn"].apply(len) >= 16].reset_index(drop=True)
    print(f"      preprocess 后可用于抽样的 prompt: {len(raw)}")

    rng = np.random.default_rng(args.seed)
    n = min(args.sample, len(raw))
    idx = rng.choice(len(raw), size=n, replace=False)
    sample = raw.iloc[idx].reset_index(drop=True)
    prompts = sample["first_turn"].tolist()

    em = LocalBGEM3Embedder(model_path=args.model_path)
    print(f"      编码 {n} 条 prompt...")
    t0 = time.time()
    vecs = em.encode_prompts(prompts)
    print(f"      编码 {time.time()-t0:.1f}s")

    strong_idx, weak_idx = model2tier[STRONG], model2tier[WEAK]

    def get_weightings(sims):
        return 10 * 10 ** (sims / np.max(sims))

    wrs = []
    t0 = time.time()
    for v in vecs:
        sims = emb @ v
        w = get_weightings(sims)
        res = compute_elo_mle_with_tie(tiered, sample_weight=w)
        ws = res[weak_idx]
        ss = res[strong_idx]
        wrs.append(float(1 - 1 / (1 + 10 ** ((ss - ws) / 400))))
    wrs = np.array(wrs)
    print(f"      计算 {time.time()-t0:.1f}s（{(time.time()-t0)/n*1000:.0f} ms/条）")

    # ---- 5) 与官方对标 ----
    print(f"\n[5/6] 与官方对标")
    print(f"  {'':14}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'全距':>10}")
    print("  " + "-" * 64)
    print(f"  {'官方 sw_ranking':14}{official['sw_ranking'].mean():>10.4f}"
          f"{official['sw_ranking'].std():>10.4f}{official['sw_ranking'].min():>10.4f}"
          f"{official['sw_ranking'].max():>10.4f}"
          f"{official['sw_ranking'].max()-official['sw_ranking'].min():>10.4f}")
    print(f"  {'本实现(含judge)':14}{wrs.mean():>10.4f}{wrs.std():>10.4f}"
          f"{wrs.min():>10.4f}{wrs.max():>10.4f}{wrs.max()-wrs.min():>10.4f}")

    print(f"\n  均值比: 本实现/官方 = {wrs.mean()/official['sw_ranking'].mean():.3f}")

    # 逐条对标（按 id 对齐）
    official_idx = official.set_index("id")
    sample_ids = sample["id"].tolist() if "id" in sample.columns else None
    if sample_ids:
        matched = official_idx.reindex(sample_ids)["sw_ranking"].values
        mask = ~np.isnan(matched)
        if mask.sum() > 10:
            corr = float(np.corrcoef(wrs[mask], matched[mask])[0, 1])
            print(f"  逐条对标（n={mask.sum()}）: corr = {corr:.4f}")
            print(f"    本实现  mean={wrs[mask].mean():.4f}")
            print(f"    官方    mean={matched[mask].mean():.4f}")

    # ---- 6) 阈值标定对比 ----
    print(f"\n[6/6] 阈值标定对比（官方 quantile 方法）")
    print(f"  {'目标占比':>10}{'官方阈值':>14}{'本实现阈值':>14}")
    for pct in [0.5, 0.3, 0.2, 0.1]:
        th_o = float(np.quantile(official["sw_ranking"], 1 - pct))
        th_m = float(np.quantile(wrs, 1 - pct))
        print(f"  {pct*100:>9.0f}%{th_o:>14.6f}{th_m:>14.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
