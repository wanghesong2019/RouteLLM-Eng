# 实验索引

每个实验记录包含：目的 → 设计 → 实测数据 → 结论 → 复现方式。
失败的实验同样保留，因为失败原因本身是有价值的结论。

| 日期 | 实验 | 状态 | 关键结论 |
|---|---|---|---|
| 2026-09-16 | [BERT 路由模型加载与区分度验证](2026-09-16-bert-router-validation.md) | ✅ 通过 | 加载正常，6.1ms/条；MMLU 上 corr(weak_acc, win_rate) = **-0.7123**（区分度强）；GSM8K 同质数据无区分度 |

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
