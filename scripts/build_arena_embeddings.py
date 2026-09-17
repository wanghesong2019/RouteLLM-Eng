#!/usr/bin/env python
"""全量生成 arena 55361 条 prompt 的 bge-m3 向量。

背景：
    替换原 generate_embeddings.py 的 OpenAI text-embedding-3-small 调用，
    改用 43 本地自托管 BAAI/bge-m3 权重。

产物：
    <out_dir>/arena_embeddings.npy       (55361, 1024) float32，L2 归一化
    <out_dir>/arena_embeddings.meta.json 元数据（条数/维度/来源/耗时/校验和）

用法：
    python scripts/build_arena_embeddings.py \
        --battles-csv <arena_train.csv> \
        --model-path  <bge-m3 本地权重目录> \
        --out-dir     <输出目录>

    例（43 部署环境）：
    python scripts/build_arena_embeddings.py \
        --battles-csv $ROUTELLM_DATA/arena_train.csv \
        --model-path  $ROUTELLM_MODELS/bge-m3 \
        --out-dir     $ROUTELLM_DATA/embeddings

自检：
    - preprocess 后行数必须是 55361（arena train.csv 的既定事实）
    - 向量 shape 必须是 (N, 1024)
    - 保存后回读校验 shape/dtype/校验和一致
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

# arena train.csv 经 MIN_LEN=16 过滤后的既定行数
EXPECTED_ARENA_ROWS = 55361


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 arena 的 bge-m3 向量")
    ap.add_argument("--battles-csv", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--expected-count", type=int, default=EXPECTED_ARENA_ROWS)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅取前 N 行做冒烟测试（0=全量）。会同步调整期望条数校验。",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, "arena_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "arena_embeddings.meta.json")

    print(f"[1/5] 读取 battles: {args.battles_csv}", flush=True)
    df = pd.read_csv(args.battles_csv)
    print(f"      原始行数: {len(df)}", flush=True)

    if args.limit and args.limit > 0:
        # 冒烟模式：截取前 N 行，期望条数随之改为实际 preprocess 后行数
        df = df.head(args.limit).copy()
        from routellm.routers.similarity_weighted.utils import preprocess_battles

        actual = preprocess_battles(df.copy()).shape[0]
        args.expected_count = actual
        print(
            f"      [冒烟模式] 截取前 {len(df)} 行 -> preprocess 后 {actual} 条", flush=True
        )

    print(f"[2/5] 加载 bge-m3 权重: {args.model_path}", flush=True)
    t_load = time.time()
    embedder = LocalBGEM3Embedder(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    embedder._ensure_model()  # 显式预热，把加载耗时单独计量
    load_sec = time.time() - t_load
    print(f"      加载完成 {load_sec:.1f}s, 维度={embedder.dimension}", flush=True)

    print(f"[3/5] 生成向量 (期望 {args.expected_count} 条)", flush=True)
    t_enc = time.time()
    vecs = embedder.encode_battles(df, expected_count=args.expected_count)
    enc_sec = time.time() - t_enc
    print(
        f"      完成 {vecs.shape} dtype={vecs.dtype} 耗时 {enc_sec:.1f}s "
        f"({vecs.shape[0] / max(enc_sec, 1e-9):.1f} 条/秒)",
        flush=True,
    )

    # ---- 自检 ----
    assert vecs.shape == (args.expected_count, BGE_M3_DIM), (
        f"shape 异常: {vecs.shape}，期望 ({args.expected_count}, {BGE_M3_DIM})"
    )
    assert vecs.dtype == np.float32, f"dtype 异常: {vecs.dtype}"
    assert np.isfinite(vecs).all(), "向量含 NaN/Inf"
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"归一化异常: norm 范围 [{norms.min():.4f}, {norms.max():.4f}]"
    )
    print(f"      自检通过: L2 norm ∈ [{norms.min():.4f}, {norms.max():.4f}]", flush=True)

    print(f"[4/5] 保存 -> {npy_path}", flush=True)
    np.save(npy_path, vecs)
    written_sha = sha256_of_file(npy_path)

    print("[5/5] 回读校验", flush=True)
    back = np.load(npy_path)
    assert back.shape == vecs.shape, f"回读 shape 不符: {back.shape} != {vecs.shape}"
    assert back.dtype == np.float32, f"回读 dtype 不符: {back.dtype}"
    assert np.array_equal(back, vecs), "回读内容与内存不一致"
    print(f"      回读一致: {back.shape} {back.dtype}", flush=True)

    meta = {
        "source_csv": os.path.abspath(args.battles_csv),
        "model_path": os.path.abspath(args.model_path),
        "model": "BAAI/bge-m3 (local, replaces OpenAI text-embedding-3-small)",
        "rows": int(vecs.shape[0]),
        "dimension": int(vecs.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "load_seconds": round(load_sec, 2),
        "encode_seconds": round(enc_sec, 2),
        "throughput_rows_per_sec": round(vecs.shape[0] / max(enc_sec, 1e-9), 2),
        "npy_sha256": written_sha,
        "npy_bytes": os.path.getsize(npy_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"      元数据 -> {meta_path}", flush=True)

    print("\n=== 全部完成 ===", flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
