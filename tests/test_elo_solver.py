"""compute_elo_mle_with_tie 求解器优化的单元测试（TDD RED 阶段）。

背景：
    实测路由延迟中 LogisticRegression.fit 占 ~90%（307ms / 110722x10）。
    数据形状是「样本极多（110k）、特征极少（10=模型数）」，
    sklearn 默认 lbfgs 对这种形状非最优；newton-cholesky 专为该形状设计。

目标：
    A) 默认使用 newton-cholesky，显著提速
    B) 结果与 lbfgs 数值一致（elo 偏差在容许范围内，不改变路由语义）
    C) 保留可配置性（需要精确复现旧结果时能切回 lbfgs）

运行：
    pytest tests/test_elo_solver.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest


def _make_arena_df(n_rows: int = 2000, n_models: int = 10, seed: int = 0) -> pd.DataFrame:
    """构造与 preprocess_battles 输出同构的 DataFrame。

    列：model_a / model_b / winner（均为已编码的 tier 整数与 winner 标签）。
    """
    rng = np.random.default_rng(seed)
    models = list(range(n_models))
    a = rng.choice(models, size=n_rows)
    b = rng.choice(models, size=n_rows)
    same = a == b
    b[same] = (b[same] + 1) % n_models  # 避免自比
    winners = rng.choice(["model_a", "model_b", "tie"], size=n_rows, p=[0.4, 0.4, 0.2])
    return pd.DataFrame({"model_a": a, "model_b": b, "winner": winners})


# ------------------------------------------------------- 1. 结果一致性


def test_result_matches_lbfgs_baseline():
    """优化后结果须与 lbfgs 基线一致 —— 判据用 **winrate 影响** 而非原始 elo 差值。

    为什么不用 elo 差值做判据：
        两求解器在小数据/病态 Hessian 下收敛到略有差异的点，
        elo 差值随数据规模浮动（实测 0.72 分 @55361 行，1.72 分 @2000 行）。
        但路由决策只取决于 winrate 与 threshold 的比较，
        因此真正该约束的是 winrate 的偏移量。

    容差推导：
        winrate = 1/(1+10^((strong-weak)/400))
        elo 差 2 分 → winrate 偏移约 0.001（千分之一），
        远小于 threshold 的典型量级（0.3~0.7），不影响路由决策。
    """
    import math

    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie
    from sklearn.linear_model import LogisticRegression

    df = _make_arena_df(2000, 10)

    optimized = compute_elo_mle_with_tie(df)

    # 手工构造 lbfgs 基线，复刻原实现
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    df2 = pd.concat([df, df], ignore_index=True)
    p, n = len(models.index), df2.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df2["model_a"]]] = +math.log(10)
    X[np.arange(n), models[df2["model_b"]]] = -math.log(10)
    Y = np.zeros(n)
    Y[df2["winner"] == "model_a"] = 1.0
    tie_idx = (df2["winner"] == "tie") | (df2["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, solver="lbfgs")
    lr.fit(X, Y)
    baseline = pd.Series(400 * lr.coef_[0] + 1000, index=models.index).sort_values(
        ascending=False
    )

    assert list(optimized.index) == list(baseline.index), "模型排序应一致"

    # 逐对模型算 winrate 偏移：这是路由决策的直接输入
    def winrate(e_strong, e_weak):
        return 1.0 / (1.0 + 10 ** ((e_strong - e_weak) / 400))

    max_wr_shift = 0.0
    for i, m_strong in enumerate(optimized.index):
        for m_weak in list(optimized.index)[i + 1 :]:
            wr_new = winrate(optimized[m_strong], optimized[m_weak])
            wr_old = winrate(baseline[m_strong], baseline[m_weak])
            max_wr_shift = max(max_wr_shift, abs(wr_new - wr_old))

    assert max_wr_shift < 0.005, (
        f"winrate 最大偏移 {max_wr_shift:.5f} 超容差（0.005）—— "
        "可能影响路由决策"
    )


def test_ranking_order_preserved():
    """模型强弱排序必须完全一致（路由语义不变）。"""
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    df = _make_arena_df(3000, 12, seed=7)
    res = compute_elo_mle_with_tie(df)
    assert list(res.values) == sorted(res.values, reverse=True), "应按 elo 降序"
    assert len(res) == 12


def test_result_with_sample_weight():
    """带 sample_weight 时（路由推理路径）结果同样合理。"""
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    df = _make_arena_df(2000, 10, seed=3)
    rng = np.random.default_rng(1)
    w = rng.uniform(1, 100, size=len(df))

    res = compute_elo_mle_with_tie(df, sample_weight=w)
    assert len(res) == 10
    assert np.isfinite(res.values).all(), "结果含 NaN/Inf"
    # 权重应实际影响结果
    res_unweighted = compute_elo_mle_with_tie(df)
    assert not np.allclose(res.values, res_unweighted.values)


# ------------------------------------------------------- 2. 可配置性


def test_solver_is_configurable():
    """须能显式切回 lbfgs（需要精确复现旧结果时的逃生舱）。"""
    import inspect

    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    sig = inspect.signature(compute_elo_mle_with_tie)
    assert "solver" in sig.parameters, "缺少 solver 参数"
    assert sig.parameters["solver"].default != "lbfgs", (
        "默认应为优化后的求解器（newton-cholesky），而非旧 lbfgs"
    )


def test_explicit_lbfgs_still_works():
    """显式传 lbfgs 时仍可正常出结果。"""
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    df = _make_arena_df(1500, 8, seed=5)
    res = compute_elo_mle_with_tie(df, solver="lbfgs")
    assert len(res) == 8
    assert np.isfinite(res.values).all()


# ------------------------------------------------------- 3. 性能


@pytest.mark.skipif(
    not (os.environ.get("ROUTELLM_ARENA_CSV") and os.environ.get("ROUTELLM_BGE_M3_PATH")),
    reason="性能对比需真实 arena 数据 + bge-m3 权重（合成权重无法复现线上的"
    "Hessian 条件，会得出相反结论）；设 ROUTELLM_ARENA_CSV / "
    "ROUTELLM_BGE_M3_PATH / ROUTELLM_ARENA_EMBEDDINGS 后启用",
)
def test_faster_than_lbfgs_on_real_data():
    """在真实路由链路上验证提速（需 ROUTELLM_ARENA_CSV + ROUTELLM_BGE_M3_PATH + 向量文件）。

    为什么必须走完整路由链路取权重：
        求解器耗时对 sample_weight 的**逐元素排列**敏感（不只是边缘分布）。
        实测：用 rng.uniform 或 beta 构造的权重，newton-cholesky 反而更慢
        （~475ms vs lbfgs ~400ms）；只有用真实路由产生的权重
        get_weightings(cosine_sims) 才复现线上的 Hessian 条件，
        newton-cholesky 才展现优势（~87ms vs ~331ms）。

    实测（55361 行）：
        Elo 求解  : lbfgs 331ms → newton-cholesky  87ms（3.8x）
        端到端路由: lbfgs 394ms → newton-cholesky 185ms（2.1x）
    """
    import time

    from routellm.routers.routers import SWRankingRouter
    from routellm.routers.similarity_weighted.utils import compute_elo_mle_with_tie

    embeddings = os.environ.get("ROUTELLM_ARENA_EMBEDDINGS")
    model_path = os.environ.get("ROUTELLM_BGE_M3_PATH")
    if not embeddings or not model_path:
        pytest.skip("需 ROUTELLM_ARENA_EMBEDDINGS 与 ROUTELLM_BGE_M3_PATH")

    router = SWRankingRouter(
        local_battles_csv=os.environ["ROUTELLM_ARENA_CSV"],
        local_embeddings_npy=embeddings,
        local_embedder_path=model_path,
    )
    emb = router._encode_prompt("What is the capital of France?")
    # 真实路由的权重：与每条 battle 的相似度精确对应
    w = router.get_weightings(np.dot(router.arena_conv_embedding, emb))

    def _time(solver):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            compute_elo_mle_with_tie(router.arena_df, sample_weight=w, solver=solver)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_lbfgs = _time("lbfgs")
    t_opt = _time(None)

    ratio = t_lbfgs / t_opt if t_opt > 0 else float("inf")
    assert ratio > 2.0, (
        f"提速不足 2 倍：opt={t_opt*1000:.1f}ms lbfgs={t_lbfgs*1000:.1f}ms "
        f"（比值 {ratio:.2f}）"
    )
