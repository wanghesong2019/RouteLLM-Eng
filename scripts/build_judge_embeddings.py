#!/usr/bin/env python
"""生成 gpt4_judge_battles 的 bge-m3 向量（补齐官方 sw_ranking 的第二个数据集）。

背景：
    官方 sw_ranking 拼接两个数据集：
        arena_battle_datasets:    [lmsys/lmsys-arena-human-preference-55k,
                                   routellm/gpt4_judge_battles]
        arena_embedding_datasets: [routellm/arena_battles_embeddings,
                                   routellm/gpt4_judge_battles_embeddings]
    此前只接了 arena 部分，导致 win_rate 与官方有 3.2 倍系统偏差
    （见 docs/experiments/2026-09-17-sw-ranking-discrimination-diagnosis.md）。
    本脚本生成缺失的 judge 部分向量。

产物：
    <out_dir>/judge_embeddings.npy       (N, 1024) float32，L2 归一化
    <out_dir>/judge_embeddings.meta.json 元数据

用法：
    python scripts/build_judge_embeddings.py \
        --battles-parquet <gpt4_judge_battles.parquet> \
        --model-path      <bge-m3 权重目录> \
        --out-dir         <输出目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routellm.routers.similarity_weighted.local_embedder import (  # noqa: E402
    BGE_M3_DIM,
    LocalBGEM3Embedder,
)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 gpt4_judge_battles 的 bge-m3 向量")
    ap.add_argument("--battles-parquet", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, "judge_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "judge_embeddings.meta.json")

    print(f"[1/5] 读取 battles: {args.battles_parquet}", flush=True)
    df = pd.read_parquet(args.battles_parquet)
    print(f"      原始行数: {len(df)}", flush=True)

    # 与 arena 同样走 preprocess_battles，保证条数语义一致
    from routellm.routers.similarity_weighted.utils import preprocess_battles

    n_proc = preprocess_battles(df.copy()).shape[0]
    print(f"      preprocess 后: {n_proc} 条", flush=True)

    print(f"[2/5] 加载 bge-m3: {args.model_path}", flush=True)
    t_load = time.time()
    embedder = LocalBGEM3Embedder(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    embedder._ensure_model()
    load_sec = time.time() - t_load
    print(f"      加载 {load_sec:.1f}s, 维度={embedder.dimension}", flush=True)

    print(f"[3/5] 生成向量（期望 {n_proc} 条）", flush=True)
    t_enc = time.time()
    vecs = embedder.encode_battles(df, expected_count=n_proc)
    enc_sec = time.time() - t_enc
    print(
        f"      完成 {vecs.shape} dtype={vecs.dtype} 耗时 {enc_sec:.1f}s "
        f"({vecs.shape[0]/max(enc_sec,1e-9):.1f} 条/秒)",
        flush=True,
    )

    # ---- 自检 ----
    assert vecs.shape == (n_proc, BGE_M3_DIM), f"shape 异常: {vecs.shape}"
    assert vecs.dtype == np.float32, f"dtype 异常: {vecs.dtype}"
    assert np.isfinite(vecs).all(), "含 NaN/Inf"
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"归一化异常: [{norms.min():.4f}, {norms.max():.4f}]"
    print(f"      自检通过: L2 norm ∈ [{norms.min():.4f}, {norms.max():.4f}]", flush=True)

    print(f"[4/5] 保存 -> {npy_path}", flush=True)
    np.save(npy_path, vecs)
    written_sha = sha256_of_file(npy_path)

    print("[5/5] 回读校验", flush=True)
    back = np.load(npy_path)
    assert back.shape == vecs.shape and np.array_equal(back, vecs), "回读不一致"
    print(f"      回读一致: {back.shape} {back.dtype}", flush=True)

    meta = {
        "source_parquet": os.path.abspath(args.battles_parquet),
        "model_path": os.path.abspath(args.model_path),
        "model": "BAAI/bge-m3",
        "rows": int(vecs.shape[0]),
        "dimension": int(vecs.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "load_seconds": round(load_sec, 2),
        "encode_seconds": round(enc_sec, 2),
        "throughput_rows_per_sec": round(vecs.shape[0] / max(enc_sec, 1e-9), 2),
        "npy_sha256": written_sha,
        "npy_bytes": os.path.getsize(npy_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("\n=== 完成 ===", flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
