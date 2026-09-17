# services/

RouteLLM 的**独立服务**，不作为 `routellm` 包的一部分。当前包含路由模型推理服务。

## inference_server.py

把 RouteLLM 的路由分类模型（BERT / CausalLLM）从 RouteLLM 进程内下沉到 host 独立服务，
RouteLLM 容器通过 HTTP 调用。设计依据见 `docs/decisions/ADR-001-inference-service-on-host.md`。

**独立实现，不 import `routellm` 包** —— 只依赖 `transformers` + `fastapi`，保持解耦。

### 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回模型加载状态与推理统计 |
| GET | `/v1/models` | 列出可用路由模型（对齐 OpenAI 格式） |
| POST | `/v1/score` | 批量计算 prompt 的 win rate（主接口） |
| GET | `/selfcheck` | 自检：用固定输入比对上游基准值 |

### win_rate 语义

```
win_rate = 1 - sum(softmax(logits)[-2:])
```

表示"应路由到强模型的程度"，与上游 `Router.route` 一致：
`win_rate >= threshold` → 强模型，否则 → 弱模型。

### 启动

```bash
python services/inference_server.py \
    --model-type bert \
    --model-path /mnt/data/wanghesong/routellm/models/bert_gpt4_augmented \
    --gpu 0 \
    --port 6070
```

### 调用示例

```bash
# 单条
curl -s -X POST localhost:6070/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"prompts": ["What is 1+1?"], "return_softmax": true}'

# 批量
curl -s -X POST localhost:6070/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"prompts": ["prompt A", "prompt B", "prompt C"]}'

# 自检（应返回 matches: true）
curl -s localhost:6070/selfcheck
```

## sw_ranking_server.py

把 **sw_ranking 路由器的完整推理链路**（bge-m3 编码 → 55k 向量相似度 → Elo 回归）
从 RouteLLM 进程内下沉到 host 独立服务，端口 **6071**。

### 为什么整条链路都下沉，而不是只下沉 bge-m3 编码

sw_ranking 需要三样东西：bge-m3 权重（2.27GB）、arena 向量（216MB）、
arena battle CSV（184MB）。若把后两样塞进容器，容器需挂载近 400MB 数据且要装
scikit-learn；若连 bge-m3 也进容器，还要装 torch —— 都会打破「675MB 轻量网关」
的设计（见 `docs/decisions/ADR-001-inference-service-on-host.md`）。
故三样全部留在 host，容器经 `host.docker.internal:6071` 调用。

**独立实现，不 import `routellm` 包** —— 只依赖 `sentence-transformers` +
`fastapi` + `scikit-learn`。Elo 回归算法为**刻意的代码重复**，一致性由
`tests/test_sw_ranking_server.py` 的数值测试与 `/selfcheck` 保障。

### 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回模型加载状态、数据规模与推理统计 |
| POST | `/v1/score` | 批量计算 prompt 的 win rate（**形状与 inference_server 的 /v1/score 一致**） |
| GET | `/selfcheck` | 自检：固定 probe 的 win_rate + top-3 相似对战 |

接口形状与 `inference_server.py` 对齐，故 RouteLLM 侧可复用同一个
`RemoteBERTRouter` 的 HTTP 逻辑（`RemoteSWRankingRouter` 直接继承它）。

### win_rate 语义

与 `inference_server.py` 及上游 `Router.route` 一致：**应路由到强模型的程度**。
`win_rate >= threshold` → 强模型，否则 → 弱模型。

（内部计算：bge-m3 编码 prompt → 与 55361 条 arena 向量算 cosine 相似度 →
`weightings = 10 * 10^(sim/max_sim)` 加权 → Elo 回归 → `strong_winrate`。）

### 启动

```bash
python services/sw_ranking_server.py \
    --model-path  <bge-m3 权重目录> \
    --battles-csv <arena_train.csv> \
    --embeddings  <arena_embeddings.npy> \
    --gpu 1 \
    --port 6071
```

前置数据准备（生成 arena 向量）见 `scripts/build_arena_embeddings.py`。

### 调用示例

```bash
# 单条
curl -s -X POST localhost:6071/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"prompts": ["What is the capital of France?"]}'

# 带诊断信息（返回 top-3 相似对战）
curl -s -X POST localhost:6071/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"prompts": ["What is the capital of France?"], "return_detail": true, "top_k": 3}'

# 自检
curl -s localhost:6071/selfcheck
```

### 实测（43 号机，RTX 4090）

| 项 | 值 |
|---|---|
| 加载耗时 | 10.2s（bge-m3 权重 + 55361 条数据 + Elo 分档） |
| 单条延迟（HTTP 端到端） | 113~141ms（p50 ≈ 114ms） |
| 批量 10 条 | 1021ms（平均 102ms/条） |
| 确定性 | 同 prompt 5 次结果完全一致 |

详细实验记录见 `docs/experiments/2026-09-17-sw-ranking-localization.md`。

## 支持状态

| model_type | 状态 | 说明 |
|---|---|---|
| `bert` | ✅ 已实现并验证 | 实测架构为 xlm-roberta，加载 1.2s，推理 6.1ms/条（RTX 4090） |
| `causal_llm` | ⏳ 接口占位 | 权重约 17GB，需处理词表末尾追加的 5 个特殊 token；待 BERT 链路验证完成后接入 |
| `sw_ranking` | ✅ 已实现并验证 | 加载 10.2s，114ms/条（HTTP 端到端），见上表 |

## 依赖

`inference_server.py`：`fastapi`、`uvicorn`、`pydantic`、`transformers`、`torch`、`numpy`

`sw_ranking_server.py`：上述基础上另需 `sentence-transformers`、`scikit-learn`、`pandas`

（均不含 `routellm` 包依赖：datasets / litellm 等不需要）
