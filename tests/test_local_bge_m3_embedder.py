"""本地 bge-m3 生成 arena 向量的单元测试（TDD RED 阶段）。

背景：
    原实现 `routellm/routers/similarity_weighted/generate_embeddings.py`
    依赖 OpenAI `text-embedding-3-small` API（每次调用付费 + 外部依赖 +
    网络抖动即失败）。改造目标：用 43 上自托管的 BAAI/bge-m3 本地权重
    生成同样的 arena 向量，去掉外部 Embedding API 依赖。

本测试聚焦改造后的本地编码器接口：
    routellm.routers.similarity_weighted.local_embedder.LocalBGEM3Embedder
        .encode_prompts(prompts: list[str]) -> np.ndarray  # (N, 1024) float32
        .dimension -> int                                  # 1024

运行：
    pytest tests/test_local_bge_m3_embedder.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest

# 本地 bge-m3 权重路径。集成测试需真实权重，通过环境变量注入，
# 避免在仓库里硬编码部署机路径（开源卫生要求）。
# 例：export ROUTELLM_BGE_M3_PATH=/path/to/BAAI--bge-m3/snapshots/master
import os

BGE_M3_PATH = os.environ.get("ROUTELLM_BGE_M3_PATH", "")

# bge-m3 dense 向量维度固定为 1024（BAAI 官方 spec）。
EXPECTED_DIM = 1024


def _make_battles_df(n_rows: int = 100) -> pd.DataFrame:
    """构造与 arena train.csv 同构的小样本 DataFrame（raw 格式）。

    注意：真实 train.csv 的 `prompt` 列是 **字符串列表** 的 JSON 编码，
    形如 ["第一轮问题","第二轮问题"]，首元素即 first_turn。
    """
    rows = []
    for i in range(n_rows):
        # 真实格式：JSON 编码的字符串列表
        prompt = [
            f"This is test prompt number {i}, long enough to pass the MIN_LEN=16 filter.",
            "A follow-up question that is deliberately long enough as well.",
        ]
        rows.append(
            {
                "id": f"test-{i}",
                "model_a": "gpt-4-0613" if i % 2 == 0 else "claude-v1",
                "model_b": "claude-v1" if i % 2 == 0 else "gpt-4-0613",
                "prompt": json.dumps(prompt),
                "winner_model_a": 1 if i % 3 == 0 else 0,
                "winner_model_b": 1 if i % 3 == 1 else 0,
                "winner_tie": 1 if i % 3 == 2 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def embedder():
    """惰性加载真实 bge-m3 权重（module 级缓存，避免重复加载 2.27GB）。"""
    if not BGE_M3_PATH or not os.path.isdir(BGE_M3_PATH):
        pytest.skip(
            "未提供本地 bge-m3 权重（设置 ROUTELLM_BGE_M3_PATH）——跳过集成测试"
        )

    from routellm.routers.similarity_weighted.local_embedder import (
        LocalBGEM3Embedder,
    )

    return LocalBGEM3Embedder(model_path=BGE_M3_PATH)


def test_module_importable_without_openai_key(monkeypatch):
    """改造后模块导入不应依赖 OPENAI_API_KEY。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib

    mod = importlib.import_module(
        "routellm.routers.similarity_weighted.local_embedder"
    )
    assert hasattr(mod, "LocalBGEM3Embedder")


def test_dimension_is_1024(embedder):
    """bge-m3 dense 维度应为 1024。"""
    assert embedder.dimension == EXPECTED_DIM


def test_encode_prompts_shape_and_dtype(embedder):
    """encode_prompts 返回 (N, 1024) 的 float32 数组。"""
    prompts = [f"Prompt {i}: explain the theory of relativity in simple terms." for i in range(100)]
    vecs = embedder.encode_prompts(prompts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (100, EXPECTED_DIM), f"期望 (100,{EXPECTED_DIM})，实际 {vecs.shape}"
    assert vecs.dtype == np.float32, f"期望 float32，实际 {vecs.dtype}"
    assert np.isfinite(vecs).all(), "向量含 NaN/Inf"


def test_embeddings_are_normalized(embedder):
    """输出应已 L2 归一化（cosine 相似度可直接点积）。"""
    prompts = ["A" * 30, "B" * 30, "C" * 30]
    vecs = embedder.encode_prompts(prompts)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"L2 范数应≈1，实际 {norms}"


def test_semantic_similarity_is_sane(embedder):
    """语义相近的句子相似度应显著高于无关句子（编码器没接反）。"""
    pairs = [
        "What is the capital of France?",
        "Which city is the capital of France?",
        "How do I bake a chocolate cake?",
    ]
    vecs = embedder.encode_prompts(pairs)
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated, (
        f"相关句相似度({sim_related:.4f}) 应高于无关句({sim_unrelated:.4f})"
    )


def test_battles_df_end_to_end(embedder):
    """从 raw battles DataFrame 到向量：条数必须与 preprocess 后行数一致。

    这是 routers.py:202 断言 `len(arena_df) == len(arena_conv_embedding)` 的根基。
    """
    df = _make_battles_df(100)
    vecs = embedder.encode_battles(df)

    assert vecs.shape[0] == 100, f"向量条数应为 100，实际 {vecs.shape[0]}"
    assert vecs.shape[1] == EXPECTED_DIM


def test_battles_count_assertion_on_length_mismatch(embedder):
    """行数不一致时必须显式报错，而非静默产出错位向量。"""
    df = _make_battles_df(50)
    with pytest.raises(AssertionError):
        embedder.encode_battles(df, expected_count=100)


def test_no_openai_dependency_at_runtime(embedder, monkeypatch):
    """本地编码全程不得触碰 OpenAI（无 key 也应成功）。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    vecs = embedder.encode_prompts(["hello world, this is a test prompt."] * 5)
    assert vecs.shape == (5, EXPECTED_DIM)
