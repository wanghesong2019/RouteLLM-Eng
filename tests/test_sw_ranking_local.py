"""SWRankingRouter 本地化改造的单元测试（TDD RED 阶段）。

改造目标（问题1 / 问题3 的一部分）：
    原 SWRankingRouter 有两处外部依赖，都在热路径或启动路径上：
      A) 构造时：load_dataset() 从 HF hub 拉 battle / embedding 数据集
      B) 推理时：calculate_strong_win_rate() 每次请求调 OpenAI
         text-embedding-3-small 生成 prompt 向量（~50ms + 计费 + 网络抖动即失败）

    改造后：
      A) 支持从本地 .npy 加载向量（arena_embedding_path）
      B) 支持用本地 bge-m3 编码 prompt（local_embedder_path）

本测试只覆盖「本地化接入」这一层，不触发真实 HF / OpenAI 网络调用。

运行：
    pytest tests/test_sw_ranking_local.py -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

BGE_M3_PATH = os.environ.get("ROUTELLM_BGE_M3_PATH", "")


# --------------------------------------------------------------- fixtures


def _make_battles_df(n_rows: int = 200) -> pd.DataFrame:
    """构造与 arena train.csv 同构的 battles DataFrame（raw 格式）。"""
    models = ["gpt-4-0613", "claude-v1", "mixtral-8x7b-instruct-v0.1", "gpt-4-1106-preview"]
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"t-{i}",
                "model_a": models[i % len(models)],
                "model_b": models[(i + 1) % len(models)],
                "prompt": json.dumps(
                    [f"Test prompt {i} padded to be longer than sixteen chars.", "follow up"]
                ),
                "winner_model_a": 1 if i % 3 == 0 else 0,
                "winner_model_b": 1 if i % 3 == 1 else 0,
                "winner_tie": 1 if i % 3 == 2 else 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def battles_csv(tmp_path):
    """把 battles 落成 CSV，供本地加载路径使用。"""
    p = tmp_path / "battles.csv"
    _make_battles_df(200).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def embeddings_npy(tmp_path):
    """造一份与 battles 条数一致的假向量矩阵 (200, 1024)，L2 归一化。"""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((200, 1024)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    p = tmp_path / "arena_embeddings.npy"
    np.save(p, vecs)
    return str(p)


# ------------------------------------------------- 1. 本地向量加载（A）


def test_construct_from_local_paths(battles_csv, embeddings_npy):
    """提供本地 battles CSV + embeddings npy 时，应能从本地构造成功。"""
    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        local_battles_csv=battles_csv,
        local_embeddings_npy=embeddings_npy,
    )
    assert router.arena_conv_embedding.shape == (200, 1024)
    assert len(router.arena_df) == 200


def test_local_load_does_not_touch_hf_hub(battles_csv, embeddings_npy, monkeypatch):
    """本地构造路径不得调用 datasets.load_dataset（避免 HF 网络依赖）。"""
    import datasets

    def _boom(*a, **k):  # pragma: no cover - 只应在被错误调用时触发
        raise AssertionError("不应调用 load_dataset（本地路径下无 HF 依赖）")

    monkeypatch.setattr(datasets, "load_dataset", _boom)

    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        local_battles_csv=battles_csv,
        local_embeddings_npy=embeddings_npy,
    )
    assert len(router.arena_df) == 200


def test_row_count_mismatch_raises(battles_csv, tmp_path):
    """向量条数与 battle 行数不一致时必须显式报错。"""
    rng = np.random.default_rng(0)
    bad = rng.standard_normal((100, 1024)).astype(np.float32)  # 应该是 200
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, bad)

    from routellm.routers.routers import SWRankingRouter

    with pytest.raises(AssertionError):
        SWRankingRouter(
            arena_battle_datasets=None,
            arena_embedding_datasets=None,
            local_battles_csv=battles_csv,
            local_embeddings_npy=str(bad_path),
        )


def test_missing_file_raises_clear_error(tmp_path):
    """文件不存在时应给出明确错误，而非静默失败。"""
    from routellm.routers.routers import SWRankingRouter

    with pytest.raises((FileNotFoundError, ValueError)):
        SWRankingRouter(
            arena_battle_datasets=None,
            arena_embedding_datasets=None,
            local_battles_csv=str(tmp_path / "nope.csv"),
            local_embeddings_npy=str(tmp_path / "nope.npy"),
        )


# ------------------------------------------------- 2. 本地 prompt 编码（B）


def test_prompt_encoding_uses_local_embedder(battles_csv, embeddings_npy):
    """配了本地编码器时，calculate_strong_win_rate 不得依赖 OpenAI。"""
    if not BGE_M3_PATH or not os.path.isdir(BGE_M3_PATH):
        pytest.skip("未提供 ROUTELLM_BGE_M3_PATH —— 跳过需要真实权重的用例")

    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        local_battles_csv=battles_csv,
        local_embeddings_npy=embeddings_npy,
        local_embedder_path=BGE_M3_PATH,
    )
    wr = router.calculate_strong_win_rate("What is the capital of France?")
    assert 0.0 <= wr <= 1.0


def test_no_openai_key_needed_when_local(battles_csv, embeddings_npy, monkeypatch):
    """本地编码器就位后，全过程不需要 OPENAI_API_KEY。"""
    if not BGE_M3_PATH or not os.path.isdir(BGE_M3_PATH):
        pytest.skip("未提供 ROUTELLM_BGE_M3_PATH —— 跳过需要真实权重的用例")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        local_battles_csv=battles_csv,
        local_embeddings_npy=embeddings_npy,
        local_embedder_path=BGE_M3_PATH,
    )
    assert 0.0 <= router.calculate_strong_win_rate("hello, a fairly long test prompt.") <= 1.0


def test_embedding_model_label_reflects_local(battles_csv, embeddings_npy):
    """embedding_model 标识应反映实际使用的模型（不再硬编码 openai 模型名）。"""
    from routellm.routers.routers import SWRankingRouter

    router = SWRankingRouter(
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        local_battles_csv=battles_csv,
        local_embeddings_npy=embeddings_npy,
    )
    assert router.embedding_model != "text-embedding-3-small"
    assert "bge-m3" in router.embedding_model


# ------------------------------------------------- 3. 向后兼容


def test_backward_compatible_signature_unchanged():
    """旧的 HF 数据集入参仍须被接受（不破坏既有调用方式）。"""
    import inspect

    from routellm.routers.routers import SWRankingRouter

    sig = inspect.signature(SWRankingRouter.__init__)
    params = set(sig.parameters)
    assert "arena_battle_datasets" in params
    assert "arena_embedding_datasets" in params
    # 新参数必须是可选的
    for name in ("local_battles_csv", "local_embeddings_npy", "local_embedder_path"):
        assert name in params, f"缺少参数 {name}"
        assert sig.parameters[name].default is None, f"{name} 应为可选（默认 None）"
