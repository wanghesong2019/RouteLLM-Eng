import abc
import functools
import random
from typing import Any

import numpy as np

# NOTE: torch / transformers / datasets / huggingface_hub 改为惰性导入。
#
# 原因：这些依赖体积巨大（torch wheel 554MB），但只有进程内推理的路由器
# （bert / causal_llm / sw_ranking / mf）需要它们。使用 remote_bert（HTTP 调用
# host 推理服务）或 random 时完全不需要。
#
# 模块级导入会强迫容器镜像包含全部重依赖 —— 见 docs/CHANGELOG.md。
# 惰性化后容器镜像从 3GB+ 降到 ~500MB，且支持"只部署网关"的部署形态。


def _lazy_import_torch():
    import torch

    return torch


def _lazy_import_transformers():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoModelForSequenceClassification, AutoTokenizer


def _lazy_import_datasets():
    from datasets import concatenate_datasets, load_dataset

    return concatenate_datasets, load_dataset


def _lazy_import_hf_hub_download():
    from huggingface_hub import hf_hub_download

    return hf_hub_download


from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)



def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        # 惰性导入：仅在使用本路由器时才需要重依赖
        from routellm.routers.causal_llm.configs import RouterModelConfig
        from routellm.routers.causal_llm.llm_utils import (
            load_prompt_format,
            to_openai_api_messages,
        )
        from routellm.routers.causal_llm.model import CausalLLMClassifier

        hf_hub_download = _lazy_import_hf_hub_download()

        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        # 惰性导入：仅在使用本路由器时才需要重依赖
        AutoModelForSequenceClassification, AutoTokenizer = _lazy_import_transformers()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
        torch = _lazy_import_torch()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    """相似度加权路由器（Elo 回归）。

    数据来源支持两种模式，互斥：
      - 远端（原行为）：arena_battle_datasets + arena_embedding_datasets，
        经 datasets.load_dataset 从 HF hub 拉取。
      - 本地（改造后）：local_battles_csv + local_embeddings_npy，
        直接从磁盘读取，无 HF 网络依赖，离线可用。

    prompt 编码同样支持两种模式：
      - 远端（原行为）：调 OpenAI text-embedding-3-small（每次请求 ~50ms + 计费）。
      - 本地（改造后）：local_embedder_path 指向本地 bge-m3 权重，
        进程内推理，零 API 成本、无网络依赖。
    """

    def __init__(
        self,
        arena_battle_datasets=None,
        arena_embedding_datasets=None,
        # This is the model pair for Elo calculations at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        num_tiers=10,
        # ---- 本地化改造新增（均可选，默认 None 时保持原远端行为）----
        local_battles_csv=None,
        local_embeddings_npy=None,
        local_embedder_path=None,
    ):
        self.strong_model = strong_model
        self.weak_model = weak_model

        use_local_data = local_battles_csv is not None or local_embeddings_npy is not None
        if use_local_data:
            if local_battles_csv is None or local_embeddings_npy is None:
                raise ValueError(
                    "本地数据模式需同时提供 local_battles_csv 与 local_embeddings_npy；"
                    f"当前收到 battles={local_battles_csv!r}, embeddings={local_embeddings_npy!r}"
                )
            self.arena_df, self.arena_conv_embedding = self._load_local_data(
                local_battles_csv, local_embeddings_npy
            )
        else:
            self.arena_df, self.arena_conv_embedding = self._load_remote_data(
                arena_battle_datasets, arena_embedding_datasets
            )

        # prompt 编码器：本地 bge-m3 优先，否则回落 OpenAI Embedding API。
        # 注意 embedding_model 需分别表达两件事：
        #   - 库内 arena 向量由什么模型生成（本地数据模式下固定 bge-m3）
        #   - 推理时 prompt 用什么编码（取决于是否给了 local_embedder_path）
        self._local_embedder = None
        self._pending_embedder_path = local_embedder_path
        if local_embedder_path is not None:
            # 编码与库向量都用本地 bge-m3
            self.embedding_model = "bge-m3"
            self.encoder_backend = "local:bge-m3"
        elif use_local_data:
            # 库向量是本地 bge-m3 生成的，但 prompt 编码仍走 OpenAI（维度不匹配会报错，
            # 属预期：本地库向量必须配本地编码器）
            self.embedding_model = "bge-m3"
            self.encoder_backend = "openai:text-embedding-3-small"
        else:
            self.embedding_model = "text-embedding-3-small"
            self.encoder_backend = "openai:text-embedding-3-small"

        assert len(self.arena_df) == len(
            self.arena_conv_embedding
        ), (
            f"Number of battle embeddings is mismatched to data: "
            f"battles={len(self.arena_df)}, embeddings={len(self.arena_conv_embedding)}"
        )

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    # ------------------------------------------------------------ 数据加载

    @staticmethod
    def _load_local_data(battles_csv, embeddings_npy):
        """从本地磁盘加载 battles 与向量，绕开 HF hub。

        Returns:
            (arena_df, embeddings)：arena_df 为 preprocess 后的 DataFrame，
            embeddings 为 (N, D) 的 float32 矩阵。
        """
        import os

        import pandas as pd

        for p in (battles_csv, embeddings_npy):
            if not os.path.exists(p):
                raise FileNotFoundError(f"本地数据文件不存在: {p}")

        battles_df = pd.read_csv(battles_csv)
        arena_df = preprocess_battles(battles_df.copy())

        embeddings = np.load(embeddings_npy)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if len(arena_df) != len(embeddings):
            raise AssertionError(
                f"向量条数 {len(embeddings)} 与 battle 行数 {len(arena_df)} 不一致，"
                "两者必须来自同一批数据（同一 preprocess 结果）"
            )
        return arena_df, embeddings

    @staticmethod
    def _load_remote_data(arena_battle_datasets, arena_embedding_datasets):
        """原行为：从 HF hub 拉取 battle 与 embedding 数据集。"""
        concatenate_datasets, load_dataset = _lazy_import_datasets()

        arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        arena_df = preprocess_battles(arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        arena_conv_embedding = np.concatenate(embeddings)
        return arena_df, arena_conv_embedding

    # ------------------------------------------------------------ 编码

    def _encode_prompt(self, prompt):
        """把 prompt 编码为向量。本地 bge-m3 优先，否则走 OpenAI Embedding API。"""
        if self._pending_embedder_path is not None:
            if self._local_embedder is None:
                from routellm.routers.similarity_weighted.local_embedder import (
                    LocalBGEM3Embedder,
                )

                self._local_embedder = LocalBGEM3Embedder(
                    model_path=self._pending_embedder_path
                )
            vecs = self._local_embedder.encode_prompts([prompt])
            return vecs[0]

        return (
            (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )

    # ------------------------------------------------------------ 推理

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        prompt_emb = self._encode_prompt(prompt)

        # 向量若已 L2 归一化（本地 bge-m3 路径），分母可省一次全量 norm 计算
        arena_norms = np.linalg.norm(self.arena_conv_embedding, axis=1)
        prompt_norm = np.linalg.norm(prompt_emb)
        if np.allclose(arena_norms, 1.0, atol=1e-3) and np.isclose(prompt_norm, 1.0, atol=1e-3):
            similarities = np.dot(self.arena_conv_embedding, prompt_emb)
        else:
            similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
                arena_norms * prompt_norm
            )

        weightings = self.get_weightings(similarities)
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        # 惰性导入：仅在使用本路由器时才需要重依赖
        from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel

        torch = _lazy_import_torch()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        del prompt
        return random.uniform(0, 1)


ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
}

# remote_* 延迟导入：remote.py 依赖本模块的 Router 基类，
# 故须在本模块定义完 Router 之后再导入，避免循环引用。
from routellm.routers.remote import (  # noqa: E402
    RemoteBERTRouter as _RemoteBERTRouter,
    RemoteSWRankingRouter as _RemoteSWRankingRouter,
)

ROUTER_CLS["remote_bert"] = _RemoteBERTRouter
ROUTER_CLS["remote_sw_ranking"] = _RemoteSWRankingRouter
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
