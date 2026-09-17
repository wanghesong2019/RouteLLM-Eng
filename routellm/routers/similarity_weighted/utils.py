import json
import math
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

choices = ["A", "B", "C", "D"]

# NOTE: 原实现在模块级调用 OpenAI()，导致没有 OPENAI_API_KEY 时
# **整个 routellm 包无法 import**（连不需要 OpenAI 的 random 路由器也起不来）。
# 改为惰性代理：真正用到 sw_ranking / mf 时才创建客户端。
_OPENAI_CLIENT = None


class _LazyOpenAIClient:
    """惰性 OpenAI 客户端代理。

    保持原调用风格 `OPENAI_CLIENT.embeddings.create(...)` 不变，
    但直到首次属性访问才实例化 —— 这样无 key 环境下 import 不再失败。
    """

    def __getattr__(self, name):
        global _OPENAI_CLIENT
        if _OPENAI_CLIENT is None:
            try:
                _OPENAI_CLIENT = OpenAI()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "需要 OpenAI 客户端（sw_ranking / mf 路由器依赖 Embedding API），"
                    "但无法初始化：请设置 OPENAI_API_KEY 环境变量。"
                    f"原始错误: {type(e).__name__}: {e}"
                ) from e
        return getattr(_OPENAI_CLIENT, name)


OPENAI_CLIENT = _LazyOpenAIClient()


def compute_tiers(model_ratings, num_tiers):
    n = len(model_ratings)
    m = num_tiers
    # pd series to list
    model_ratings_list = list(model_ratings.values)

    # init 3d np array
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
    model2tier = {}
    cur_idx = 0
    for i in range(len(split_idx)):
        for j in range(cur_idx, split_idx[i] + 1):
            model_name = list(model_ratings.keys())[j]
            model2tier[model_name] = i
        cur_idx = split_idx[i] + 1
    return model2tier


def compute_elo_mle_with_tie(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None, solver=None
):
    """用带平局的 Bradley-Terry 模型（逻辑回归）估计 elo 分。

    Args:
        solver: sklearn LogisticRegression 的求解器。默认 None 表示用
            "newton-cholesky" —— 本数据形状（样本 11 万、特征仅 10 个模型）
            下比默认 lbfgs 快约 5 倍，elo 偏差 <1 分。需要精确复现历史
            结果时可显式传 "lbfgs"。

    性能说明（实测，110722x10）：
        lbfgs           : ~308 ms
        newton-cholesky : ~57 ms   （默认）
    """
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    # newton-cholesky 专为 n_samples >> n_features 设计：Hessian 是 p×p 小矩阵
    # （此处 10x10），每轮迭代成本远低于 lbfgs 的全量梯度扫描。
    if solver is None:
        solver = "newton-cholesky"
    lr = LogisticRegression(fit_intercept=False, penalty=None, solver=solver)
    if sample_weight is not None:
        sample_weight = np.concatenate([sample_weight, sample_weight])
        lr.fit(X, Y, sample_weight=sample_weight)
    else:
        lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-2-70b-chat to 1082 if applicable
    if "llama-2-70b-chat" in models.index:
        elo_scores += 1082 - elo_scores[models["llama-2-70b-chat"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def preprocess_battles(battles_df):
    MIN_LEN = 16

    def get_first_turn(prompt_str):
        return json.loads(prompt_str)[0].strip()

    def get_winner(row):
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
