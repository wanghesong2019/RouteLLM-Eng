# 实验索引

每个实验记录包含：目的 → 设计 → 实测数据 → 结论 → 复现方式。
失败的实验同样保留，因为失败原因本身是有价值的结论。

| 日期 | 实验 | 状态 | 关键结论 |
|---|---|---|---|
| 2026-09-16 | [BERT 路由模型加载与区分度验证](2026-09-16-bert-router-validation.md) | ✅ 通过 | 加载正常，6.1ms/条；MMLU 上 corr(weak_acc, win_rate) = **-0.7123**（区分度强）；GSM8K 同质数据无区分度 |
| 2026-09-16 | [推理服务 API 端到端验证](2026-09-16-inference-api-validation.md) | ✅ 通过 | 与上游逐位一致（`matches: true`）；HTTP 开销 4.6ms；batch=128 加速 **68x**，MMLU 14000 题 1.3s |
| 2026-09-16 | [RouteLLM 与推理服务链路打通](2026-09-16-remote-router-e2e.md) | ✅ 通过 | 新增 `RemoteBERTRouter`（19 个单测全绿）；跨机器 win_rate 逐位一致；**发现 43 仅开放 SSH，Docker 化须两者同在 43** |
| 2026-09-16 | [Docker 化部署与全链路验证](2026-09-16-docker-deployment-e2e.md) | ✅ 通过 | 镜像 675MB（无 torch）；容器经 `host.docker.internal` 调用 host 推理服务；网关→路由→下游 LLM 全链路打通 |
| 2026-09-16 | [环境探查：33/43/A6000 三机对比](2026-09-16-environment-survey.md) | 决策 | 维持 43 方案（唯一 GPU 空闲的机器）；**A6000 即 GitLab 主机**（172.17.17.50）；33 与 A6000 的 GPU 均被 vLLM 占满 |

## 结论摘要（供快速引用）

### BERT 路由器（`routellm/bert_gpt4_augmented`）

- 实际架构 **xlm-roberta**，3 标签（上游命名为 `bert`）
- 加载 1.2s，单条推理 6.1ms（RTX 4090）
- `win_rate = 1 - sum(softmax(logits)[-2:])`，语义为"应路由到强模型的程度"
- **57 学科 MMLU 上 corr = -0.7123** —— 弱模型越不擅长的学科，win rate 越高
- 适用边界：依赖"query 文本能反映难度"。GSM8K 这类同质化数据集上无区分度

### 环境版本（33 号机 venv）

```
Python 3.10.12 / torch 2.14.0 / transformers 5.17.0 / datasets 5.0.1 / litellm 1.101.0
```

`import` 全部通过，`transformers 5.x` 未破坏导入。
