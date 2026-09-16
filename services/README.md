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

## 支持状态

| model_type | 状态 | 说明 |
|---|---|---|
| `bert` | ✅ 已实现并验证 | 实测架构为 xlm-roberta，加载 1.2s，推理 6.1ms/条（RTX 4090） |
| `causal_llm` | ⏳ 接口占位 | 权重约 17GB，需处理词表末尾追加的 5 个特殊 token；待 BERT 链路验证完成后接入 |

## 依赖

`fastapi`、`uvicorn`、`pydantic`、`transformers`、`torch`、`numpy`

（不含 `routellm` 包依赖：datasets / litellm / scikit-learn 等均不需要）
