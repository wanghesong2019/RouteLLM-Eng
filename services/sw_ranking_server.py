"""sw_ranking 路由推理服务（host 侧）。

把 sw_ranking 的完整推理链路（bge-m3 编码 → 相似度 → Elo 回归）从 RouteLLM
进程内下沉到 host 独立服务，RouteLLM 容器通过 HTTP 调用。设计见
docs/decisions/ADR-001（模型推理下沉到 host）。

为什么整个 sw_ranking 都下沉，而不是只下沉 bge-m3 编码
--------------------------------------------------------
sw_ranking 需要三样东西：bge-m3 权重（2.27GB）、arena 向量（216MB）、
arena battle CSV（184MB）。若把后两样塞进容器，容器需挂载近 400MB 数据
且要装 scikit-learn；若连 bge-m3 也进容器，还要装 torch。
这都会打破「675MB 轻量网关」的设计。
故：三样全部留在 host，容器经 host.docker.internal 调用本服务，
接口与 services/inference_server.py 的 /v1/score 完全对齐 ——
容器侧只需把路由名从 remote_bert 换成 remote_sw_ranking，保持零挂载。

独立实现，不 import routellm 包 —— 只依赖 sentence-transformers + fastapi
+ scikit-learn，保持解耦（与 inference_server.py 同样的约定）。

接口设计依据
------------
- win_rate 语义与上游 SWRankingRouter 一致：strong_winrate ∈ [0,1]，
  >= threshold 则路由到强模型
- 支持 batch：arena 评测需批量处理
- return_detail 可选返回 top-k 相似对战，用于诊断路由依据

启动
----
    python services/sw_ranking_server.py \
        --model-path  <bge-m3 权重目录> \
        --battles-csv <arena_train.csv> \
        --embeddings  <arena_embeddings.npy> \
        --port 6071

自检
----
    curl -s localhost:6071/selfcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("routellm.sw_ranking")

# ---------------------------------------------------------------------------
# 全局状态（进程内单例）
# ---------------------------------------------------------------------------
EMBEDDER: Optional[Any] = None
ARENA_DF: Optional[pd.DataFrame] = None
ARENA_EMB: Optional[np.ndarray] = None
MODEL2TIER: Optional[Dict[Any, int]] = None
CONFIG: Dict[str, Any] = {}
START_TS: float = 0.0

SCORE_COUNT: int = 0
SCORE_TOTAL_MS: float = 0.0

STRONG_MODEL = "gpt-4-1106-preview"
WEAK_MODEL = "mixtral-8x7b-instruct-v0.1"
NUM_TIERS = 10

# 上游 preprocess_battles 的 MIN_LEN（复刻，保持行为一致）
MIN_LEN = 16


# ---------------------------------------------------------------------------
# 请求 / 响应模型（与 inference_server.py 的 /v1/score 形状对齐）
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, description="待评分 prompt 列表")
    return_detail: bool = Field(False, description="是否返回 top-k 相似对战（诊断用）")
    top_k: int = Field(3, ge=1, le=20, description="return_detail 时的 top-k")


class ScoreResult(BaseModel):
    win_rate: float
    top_similar: Optional[List[Dict[str, Any]]] = None


class ScoreResponse(BaseModel):
    results: List[ScoreResult]
    model_type: str
    count: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# 算法复刻（与 routellm.routers.similarity_weighted.utils 保持一致）
#
# 这里是刻意的代码重复：服务须独立部署、不 import routellm 包。
# 一致性由 tests/test_sw_ranking_server.py 的数值测试 + /selfcheck 保障。
# ---------------------------------------------------------------------------
def preprocess_battles(battles_df: pd.DataFrame) -> pd.DataFrame:
    """复刻上游 preprocess_battles。"""

    def get_first_turn(prompt_str: str) -> str:
        return json.loads(prompt_str)[0].strip()

    def get_winner(row: pd.Series) -> str:
        if row["winner_model_a"] == 1:
            return "model_a"
        elif row["winner_model_b"] == 1:
            return "model_b"
        else:
            return "tie"

    battles_df["first_turn"] = battles_df["prompt"].apply(get_first_turn)
    battles_df["winner"] = battles_df.apply(get_winner, axis=1)
    battles_df = battles_df.loc[battles_df["first_turn"].apply(len) >= MIN_LEN]
    battles_df = battles_df[["model_a", "model_b", "winner"]]
    return battles_df


def compute_elo_mle_with_tie(
    df: pd.DataFrame,
    SCALE: float = 400,
    BASE: int = 10,
    INIT_RATING: float = 1000,
    sample_weight: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
) -> pd.Series:
    """复刻上游 compute_elo_mle_with_tie（含 newton-cholesky 优化）。

    solver 默认 newton-cholesky：本数据形状（11 万行 × 10 模型）下比 lbfgs
    快约 4 倍，elo 偏差 <1 分（见 docs/experiments/2026-09-17-sw-ranking-localization.md）。
    """
    from sklearn.linear_model import LogisticRegression

    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    if solver is None:
        solver = "newton-cholesky"
    lr = LogisticRegression(fit_intercept=False, penalty=None, solver=solver)
    if sample_weight is not None:
        sample_weight = np.concatenate([sample_weight, sample_weight])
        lr.fit(X, Y, sample_weight=sample_weight)
    else:
        lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "llama-2-70b-chat" in models.index:
        elo_scores += 1082 - elo_scores[models["llama-2-70b-chat"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compute_tiers(model_ratings: pd.Series, num_tiers: int) -> Dict[Any, int]:
    """复刻上游 compute_tiers（动态规划分档）。"""
    n = len(model_ratings)
    m = num_tiers
    model_ratings_list = list(model_ratings.values)

    dp = np.zeros((n, n, m))
    dp_split = np.zeros((n, n, m))
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j][0] = np.var(model_ratings_list[i : j + 1])

    for tier in range(1, m):
        for i in range(n):
            for j in range(i + 1, n):
                dp[i][j][tier] = 1000000000
                for l in range(i, j):
                    if dp[i][j][tier] > dp[i][l][tier - 1] + dp[l + 1][j][0]:
                        dp_split[i][j][tier] = l
                        dp[i][j][tier] = dp[i][l][tier - 1] + dp[l + 1][j][0]

    cur_n = n
    split_idx = []
    for tier in range(m - 1, 0, -1):
        split = int(dp_split[0][cur_n - 1][tier])
        split_idx.append(split)
        cur_n = split + 1

    split_idx = split_idx[::-1] + [n - 1]
    model2tier: Dict[Any, int] = {}
    cur_idx = 0
    for i in range(len(split_idx)):
        for j in range(cur_idx, split_idx[i] + 1):
            model_name = list(model_ratings.keys())[j]
            model2tier[model_name] = i
        cur_idx = split_idx[i] + 1
    return model2tier


# ---------------------------------------------------------------------------
# 模型 / 数据加载
# ---------------------------------------------------------------------------
def load_router(
    model_path: str,
    battles_csv: str,
    embeddings_npy: str,
    judge_parquet: Optional[str] = None,
    judge_embeddings: Optional[str] = None,
) -> None:
    """加载 bge-m3 + arena 数据（可选拼接 judge 数据集），并预计算 Elo 分档。

    官方 sw_ranking 拼接两个数据集：
        arena (lmsys/...) + judge (routellm/gpt4_judge_battles)
    只接 arena 会与官方产生系统偏差（实测 mean 0.692 vs 0.216），
    因此生产环境应同时提供 judge_parquet 与 judge_embeddings。
    """
    global EMBEDDER, ARENA_DF, ARENA_EMB, MODEL2TIER, CONFIG, START_TS

    paths = [("bge-m3 权重", model_path), ("arena CSV", battles_csv),
             ("arena 向量", embeddings_npy)]
    if judge_parquet:
        paths.append(("judge parquet", judge_parquet))
    if judge_embeddings:
        paths.append(("judge 向量", judge_embeddings))
    for label, p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} 不存在: {p}")

    t0 = time.time()

    logger.info("加载 bge-m3: %s", model_path)
    from sentence_transformers import SentenceTransformer

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_path, device=device)
    embedder.max_seq_length = 512

    logger.info("加载 arena 数据: %s", battles_csv)
    arena_df = preprocess_battles(pd.read_csv(battles_csv))
    arena_emb = np.load(embeddings_npy).astype(np.float32)
    if len(arena_df) != len(arena_emb):
        raise ValueError(
            f"arena 向量条数 {len(arena_emb)} 与 battle 行数 {len(arena_df)} 不一致"
        )

    datasets_info = [{"name": "arena", "rows": int(len(arena_df))}]

    # 拼接 judge 数据集（官方配置的第二部分）
    if judge_parquet and judge_embeddings:
        logger.info("加载 judge 数据: %s", judge_parquet)
        judge_df = preprocess_battles(pd.read_parquet(judge_parquet))
        judge_emb = np.load(judge_embeddings).astype(np.float32)
        if len(judge_df) != len(judge_emb):
            raise ValueError(
                f"judge 向量条数 {len(judge_emb)} 与 battle 行数 {len(judge_df)} 不一致"
            )
        logger.info("拼接: arena %d + judge %d", len(arena_df), len(judge_df))
        arena_df = pd.concat([arena_df, judge_df], ignore_index=True)
        arena_emb = np.concatenate([arena_emb, judge_emb], axis=0)
        datasets_info.append({"name": "judge", "rows": int(len(judge_df))})

    logger.info("计算 Elo 分档（%s 条）", len(arena_df))
    model_ratings = compute_elo_mle_with_tie(arena_df)
    model2tier = compute_tiers(model_ratings, num_tiers=NUM_TIERS)

    arena_df = arena_df.copy()
    arena_df["model_a"] = arena_df["model_a"].apply(lambda x: model2tier[x])
    arena_df["model_b"] = arena_df["model_b"].apply(lambda x: model2tier[x])

    EMBEDDER = embedder
    ARENA_DF = arena_df
    ARENA_EMB = arena_emb
    MODEL2TIER = model2tier
    START_TS = time.time()

    CONFIG = {
        "model_type": "sw_ranking",
        "model_path": model_path,
        "embedding_model": "BAAI/bge-m3",
        "device": device,
        "rows": int(len(arena_df)),
        "dimension": int(arena_emb.shape[1]),
        "datasets": datasets_info,
        "num_tiers": NUM_TIERS,
        "strong_model": STRONG_MODEL,
        "weak_model": WEAK_MODEL,
        "load_seconds": round(time.time() - t0, 2),
    }
    logger.info("加载完成 %.1fs: %s", time.time() - t0, CONFIG)


# ---------------------------------------------------------------------------
# 核心推理
# ---------------------------------------------------------------------------
def _get_weightings(similarities: np.ndarray) -> np.ndarray:
    """复刻上游 get_weightings：10 * 10^(sim/max_sim)。"""
    max_sim = np.max(similarities)
    return 10 * 10 ** (similarities / max_sim)


def score_batch(
    prompts: List[str], return_detail: bool = False, top_k: int = 3
) -> List[Dict[str, Any]]:
    """计算 prompt 的 strong win rate。

    win_rate 语义：应路由到强模型的程度（与上游 Router.route 一致）。
    """
    if EMBEDDER is None or ARENA_DF is None or ARENA_EMB is None:
        raise RuntimeError("model not loaded")

    vecs = EMBEDDER.encode(
        prompts,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    out: List[Dict[str, Any]] = []
    for i, vec in enumerate(vecs):
        # 向量已 L2 归一化 → cosine 相似度即点积
        sims = ARENA_EMB @ vec

        weightings = _get_weightings(sims)
        res = compute_elo_mle_with_tie(ARENA_DF, sample_weight=weightings)

        weak_score = res[MODEL2TIER[WEAK_MODEL]]
        strong_score = res[MODEL2TIER[STRONG_MODEL]]
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = float(1 - weak_winrate)

        item: Dict[str, Any] = {"win_rate": strong_winrate}

        if return_detail:
            idx = np.argsort(sims)[::-1][:top_k]
            item["top_similar"] = [
                {
                    "rank": int(r + 1),
                    "similarity": float(sims[j]),
                    "weighting": float(weightings[j]),
                    "model_a_tier": int(ARENA_DF.iloc[j]["model_a"]),
                    "model_b_tier": int(ARENA_DF.iloc[j]["model_b"]),
                    "winner": str(ARENA_DF.iloc[j]["winner"]),
                }
                for r, j in enumerate(idx)
            ]
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RouteLLM sw_ranking Inference Service",
    description="sw_ranking（bge-m3 + Elo 回归）路由推理服务（host 侧）",
    version="0.1.0",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查。模型未加载时返回 not_ready，供容器启动探针使用。"""
    ready = EMBEDDER is not None
    return {
        "status": "online" if ready else "not_ready",
        "model_loaded": ready,
        "uptime_seconds": round(time.time() - START_TS, 1) if ready else 0,
        **CONFIG,
        "stats": {
            "score_count": SCORE_COUNT,
            "avg_ms": round(SCORE_TOTAL_MS / SCORE_COUNT, 2) if SCORE_COUNT else 0.0,
        },
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> Dict[str, Any]:
    """批量计算 prompt 的 win rate（形状与 BERT 服务的 /v1/score 对齐）。"""
    global SCORE_COUNT, SCORE_TOTAL_MS

    if EMBEDDER is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    t = time.time()
    try:
        results = score_batch(req.prompts, req.return_detail, req.top_k)
    except Exception as e:  # noqa: BLE001
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference error: {e}") from e

    elapsed_ms = (time.time() - t) * 1000
    SCORE_COUNT += len(req.prompts)
    SCORE_TOTAL_MS += elapsed_ms

    return {
        "results": results,
        "model_type": "sw_ranking",
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 2),
    }


@app.get("/selfcheck")
def selfcheck() -> Dict[str, Any]:
    """自检：用固定输入确认服务可用，并输出可跨版本比对的 win_rate。"""
    if EMBEDDER is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    probe = "What is the capital of France?"
    res = score_batch([probe], return_detail=True, top_k=3)[0]
    return {
        "probe": probe,
        "win_rate": res["win_rate"],
        "top_similar": res["top_similar"],
        "note": "基准值见 docs/experiments/2026-09-17-sw-ranking-localization.md",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RouteLLM sw_ranking 推理服务")
    ap.add_argument("--model-path", required=True, help="bge-m3 权重目录")
    ap.add_argument("--battles-csv", required=True, help="arena battle CSV")
    ap.add_argument("--embeddings", required=True, help="arena 向量 .npy")
    ap.add_argument("--judge-parquet", default=None,
                    help="gpt4_judge_battles parquet（官方配置的第二数据集，建议提供）")
    ap.add_argument("--judge-embeddings", default=None,
                    help="gpt4_judge_battles 向量 .npy（与 --judge-parquet 配对）")
    ap.add_argument("--port", type=int, default=6071)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gpu", type=int, default=None, help="指定 GPU 序号")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if bool(args.judge_parquet) != bool(args.judge_embeddings):
        raise SystemExit(
            "--judge-parquet 与 --judge-embeddings 须成对提供（两者缺一会导致"
            "只接入部分数据集，与官方产生系统偏差）"
        )

    load_router(
        args.model_path,
        args.battles_csv,
        args.embeddings,
        judge_parquet=args.judge_parquet,
        judge_embeddings=args.judge_embeddings,
    )
    logger.info("starting server on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
