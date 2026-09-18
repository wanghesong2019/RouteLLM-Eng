"""本地 bge-m3 向量生成器（替换 OpenAI Embedding API）。

改造动机：
    原 `generate_embeddings.py` 调用 OpenAI `text-embedding-3-small`：
      - 每次请求外部 API（成本 + 延迟）
      - 需要 OPENAI_API_KEY（环境耦合）
      - 网络抖动/限流即失败（无容错）
    改为加载自托管的 BAAI/bge-m3 本地权重，推理在本机 GPU 完成。

设计：
    - 惰性加载：模型只在首次 encode 时进显存，避免 import 即占 2.3GB。
    - L2 归一化输出：cosine 相似度退化为点积，与 sw_ranking 的
      相似度计算路径天然对齐。
    - encode_battles() 复用 preprocess_battles()，保证向量条数与
      routers.py 里 `assert len(arena_df) == len(arena_conv_embedding)`
      所需的语义完全一致（都是 preprocess 后的行数）。

维度说明：
    bge-m3 dense 输出固定 1024 维，与原 text-embedding-3-small 的 1536 维
    不同。routers.py 只做 len() 一致性断言、未写死维度，故可直接替换。
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from routellm.routers.similarity_weighted.utils import preprocess_battles

logger = logging.getLogger(__name__)

# bge-m3 dense 维度（BAAI 官方 spec）
BGE_M3_DIM = 1024


class LocalBGEM3Embedder:
    """基于本地 bge-m3 权重的向量生成器。

    Args:
        model_path: 本地权重目录（含 config.json / pytorch_model.bin /
            modules.json 等 sentence-transformers 格式文件）。
        device: 推理设备。"auto" 自动选 cuda，不可用则 cpu。
        batch_size: 单批编码条数。
        max_length: 截断长度，bge-m3 支持 8192，默认 512 兼顾速度。
        normalize: 是否 L2 归一化，默认 True。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._model = None  # 惰性加载
        self._dimension = BGE_M3_DIM

    # ---------------------------------------------------------------- load

    def _ensure_model(self) -> None:
        """首次调用时加载权重到目标设备。"""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(
            "加载本地 bge-m3 权重 path=%s device=%s", self.model_path, self.device
        )
        model = SentenceTransformer(self.model_path, device=self.device)
        model.max_seq_length = self.max_length
        self._model = model

        # 以实际模型输出为准校正维度（防御权重与预期不符）
        actual = model.get_sentence_embedding_dimension()
        if actual != self._dimension:
            logger.warning(
                "bge-m3 实际维度 %s 与预期 %s 不符，以实际为准", actual, self._dimension
            )
            self._dimension = actual

    # ------------------------------------------------------------ properties

    @property
    def dimension(self) -> int:
        """dense 向量维度。未加载时返回 bge-m3 的官方 spec 值 1024。"""
        return self._dimension

    # --------------------------------------------------------------- encode

    def encode_prompts(self, prompts: Sequence[str]) -> np.ndarray:
        """把文本列表编码为 (N, dimension) 的 float32 数组。

        Args:
            prompts: 文本列表。

        Returns:
            np.ndarray，shape (N, 1024)，dtype float32。normalize=True 时
            每行已 L2 归一化。
        """
        if len(prompts) == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)

        self._ensure_model()

        vecs = self._model.encode(
            list(prompts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(prompts) > 1000,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        assert vecs.shape[1] == self._dimension, (
            f"编码结果维度 {vecs.shape[1]} 与期望 {self._dimension} 不符"
        )
        return vecs

    def encode_battles(
        self,
        battles_df: Any,
        expected_count: Optional[int] = None,
    ) -> np.ndarray:
        """从 raw arena battles DataFrame 生成向量矩阵。

        内部走 preprocess_battles()，与 routers.py 的 arena_df 行数语义一致
        （即 MIN_LEN=16 过滤后的行数）。

        Args:
            battles_df: raw battles DataFrame，须含 prompt / model_a /
                model_b / winner_model_a / winner_model_b 列。
            expected_count: 期望的向量条数；给出时严格校验，不一致即报错，
                避免向量与 arena_df 错位（routers.py:202 的断言根基）。

        Returns:
            np.ndarray，shape (N, 1024)。
        """
        processed = preprocess_battles(battles_df.copy())
        n = processed.shape[0]

        if expected_count is not None:
            assert n == expected_count, (
                f"preprocess 后行数 {n} 与期望 {expected_count} 不一致，"
                "向量将与 arena_df 错位"
            )

        # preprocess_battles 丢弃了 first_turn 列，需按同一过滤逻辑重新取文本。
        # 关键：不能重复硬编码 MIN_LEN，否则与 preprocess 的实现漂移会导致
        # 文本条数与行数错位。这里用「解析全部 -> 同 preprocess 一样过滤」的方式，
        # 并通过 processed.index 对齐，保证与 preprocess 的保留行完全一致。
        import json

        raw = battles_df.copy()
        raw["first_turn"] = raw["prompt"].apply(lambda s: json.loads(s)[0].strip())
        # 用 index 对齐取文本，而非重复长度阈值判断
        first_turns: List[str] = raw.loc[processed.index, "first_turn"].tolist()

        assert len(first_turns) == n, (
            f"文本条数 {len(first_turns)} 与 preprocess 行数 {n} 不一致"
        )

        logger.info("开始生成 %s 条 arena 向量（batch=%s）", n, self.batch_size)
        vecs = self.encode_prompts(first_turns)

        if expected_count is not None:
            assert vecs.shape[0] == expected_count, (
                f"向量条数 {vecs.shape[0]} 与期望 {expected_count} 不一致"
            )

        return vecs

    def save(self, vecs: np.ndarray, out_path: str) -> None:
        """把向量矩阵存为 .npy（float32）。"""
        assert vecs.dtype == np.float32, f"仅保存 float32，实际 {vecs.dtype}"
        np.save(out_path, vecs)
        logger.info("已保存 %s 条向量 -> %s", vecs.shape[0], out_path)
