"""SWRankingRouter 多数据集拼接的单元测试（TDD RED 阶段）。

背景：
    官方 sw_ranking 拼接两个数据集：
        arena (lmsys/lmsys-arena-human-preference-55k)  ← 已接入
      + judge (routellm/gpt4_judge_battles)             ← 缺失，导致系统偏差

    实测证据（docs/experiments/2026-09-17-sw-ranking-discrimination-diagnosis.md）：
        仅 arena :  mean=0.6919  corr(官方)=0.076   ← 与官方差 3.2 倍
        arena+judge: mean=0.2155  corr(官方)=0.766  ← 均值比 0.995

    故需让 SWRankingRouter 支持传入多个本地数据集并正确拼接。

设计：
    现有单数据集参数（local_battles_csv / local_embeddings_npy）保留不变，
    新增 list 形式参数支持拼接。拼接顺序必须一致：
        battles 顺序 == embeddings 顺序（否则向量与对战记录错位）

运行：
    pytest tests/test_sw_ranking_multi_dataset.py -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest


def _make_df(n_rows: int, seed: int, id_prefix: str) -> pd.DataFrame:
    """构造与 arena/judge 同构的 battles DataFrame。"""
    rng = np.random.default_rng(seed)
    models = ["gpt-4-0613", "claude-v1", "mixtral-8x7b-instruct-v0.1"]
    rows = []
    for i in range(n_rows):
        rows.append({
            "id": f"{id_prefix}-{i}",
            "model_a": models[i % len(models)],
            "model_b": models[(i + 1) % len(models)],
            "prompt": json.dumps([f"Prompt {id_prefix}-{i} padded beyond sixteen chars.", "follow"]),
            "winner_model_a": 1 if i % 3 == 0 else 0,
            "winner_model_b": 1 if i % 3 == 1 else 0,
            "winner_tie": 1 if i % 3 == 2 else 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def two_datasets(tmp_path):
    """两个数据集（模拟 arena + judge）+ 对应向量。"""
    out = []
    for name, n, seed in [("arena", 200, 0), ("judge", 300, 1)]:
        csv = tmp_path / f"{name}.csv"
        _make_df(n, seed, name).to_csv(csv, index=False)

        rng = np.random.default_rng(seed + 100)
        vecs = rng.standard_normal((n, 1024)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        npy = tmp_path / f"{name}.npy"
        np.save(npy, vecs)
        out.append({"battles": str(csv), "embeddings": str(npy), "n": n})
    return out


# ------------------------------------------------- 拼接正确性


def test_multi_dataset_concatenates_rows(two_datasets):
    """传入多个数据集时，行数应为各数据集之和。"""
    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        local_datasets=two_datasets,
    )
    assert len(router.arena_df) == 500, f"应为 200+300=500，实际 {len(router.arena_df)}"
    assert router.arena_conv_embedding.shape == (500, 1024)


def test_multi_dataset_order_matters(tmp_path):
    """battles 与 embeddings 的拼接顺序必须一致，否则应报错。

    构造：数据集 A 的 battles 配 数据集 B 的 embeddings（行数不同）→ 必须报错。
    """
    from routellm.routers.routers import SWRankingRouter

    a = _make_df(100, 0, "a")
    b = _make_df(150, 1, "b")
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)

    rng = np.random.default_rng(0)
    va = rng.standard_normal((100, 1024)).astype(np.float32)
    vb = rng.standard_normal((150, 1024)).astype(np.float32)
    np.save(tmp_path / "a.npy", va)
    np.save(tmp_path / "b.npy", vb)

    # 总行数对得上（250），但配对错误：A 的 battles 配 B 的向量
    # 这种错误在总量层面无法察觉，故应通过 per-dataset 校验拦截
    bad = [
        {"battles": str(tmp_path / "a.csv"), "embeddings": str(tmp_path / "b.npy")},
        {"battles": str(tmp_path / "b.csv"), "embeddings": str(tmp_path / "a.npy")},
    ]
    with pytest.raises((AssertionError, ValueError)):
        SWRankingRouter(local_datasets=bad)


def test_multi_dataset_per_dataset_count_check(two_datasets):
    """每个数据集自身须满足 battles 行数 == 向量条数。

    用实现认识的 'count' 字段声明一个错误的条数（实际 200，声明 999），
    应被 per-dataset 校验拦截。这用于防止 battles/embeddings 错配。
    """
    from routellm.routers.routers import SWRankingRouter

    ds = [dict(d) for d in two_datasets]
    ds[0] = dict(ds[0])
    ds[0]["count"] = 999  # 声明错误条数（实际 preprocess 后 200）

    with pytest.raises((AssertionError, ValueError)):
        SWRankingRouter(local_datasets=ds)


def test_multi_dataset_swapped_embeddings_raises(tmp_path):
    """battles 与 embeddings 跨数据集错配时须报错（per-dataset 校验拦截）。

    这是比「总量相等」更强的校验：A 的 battles 配 B 的 embeddings，
    总量可能仍对得上，但 per-dataset 校验能发现。
    """
    from routellm.routers.routers import SWRankingRouter

    a = _make_df(100, 0, "a")
    b = _make_df(150, 1, "b")
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)

    rng = np.random.default_rng(0)
    np.save(tmp_path / "a.npy", rng.standard_normal((100, 1024)).astype(np.float32))
    np.save(tmp_path / "b.npy", rng.standard_normal((150, 1024)).astype(np.float32))

    swapped = [
        {"battles": str(tmp_path / "a.csv"), "embeddings": str(tmp_path / "b.npy")},
        {"battles": str(tmp_path / "b.csv"), "embeddings": str(tmp_path / "a.npy")},
    ]
    with pytest.raises((AssertionError, ValueError)):
        SWRankingRouter(local_datasets=swapped)


# ------------------------------------------------- 单数据集兼容


def test_single_dataset_still_works(two_datasets):
    """旧的单数据集参数仍须可用（向后兼容）。"""
    from routellm.routers.routers import SWRankingRouter

    d = two_datasets[0]
    router = SWRankingRouter(
        local_battles_csv=d["battles"],
        local_embeddings_npy=d["embeddings"],
    )
    assert len(router.arena_df) == 200


def test_multi_equivalent_to_single_when_one_dataset(two_datasets):
    """只传一个数据集时，结果应与单数据集模式等价。"""
    from routellm.routers.routers import SWRankingRouter

    d = two_datasets[0]
    r1 = SWRankingRouter(
        local_datasets=[{"battles": d["battles"], "embeddings": d["embeddings"]}]
    )
    r2 = SWRankingRouter(
        local_battles_csv=d["battles"], local_embeddings_npy=d["embeddings"]
    )
    assert len(r1.arena_df) == len(r2.arena_df)
    assert np.allclose(r1.arena_conv_embedding, r2.arena_conv_embedding)


# ------------------------------------------------- 推理可用


def test_multi_dataset_can_score(two_datasets):
    """多数据集构造后应能正常推理（需本地编码器，否则会回落 OpenAI）。"""
    from routellm.routers.routers import SWRankingRouter

    bge = os.environ.get("ROUTELLM_BGE_M3_PATH", "")
    if not bge or not os.path.isdir(bge):
        pytest.skip("需 ROUTELLM_BGE_M3_PATH 才能测端到端评分")

    router = SWRankingRouter(local_datasets=two_datasets, local_embedder_path=bge)
    wr = router.calculate_strong_win_rate("What is the capital of France?")
    assert 0.0 <= wr <= 1.0


def test_without_encoder_requires_openai(two_datasets):
    """未配本地编码器时，编码会尝试 OpenAI —— 须给出明确错误而非静默失败。"""
    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(local_datasets=two_datasets)
    assert router.encoder_backend.startswith("openai:")
    assert router._pending_embedder_path is None
