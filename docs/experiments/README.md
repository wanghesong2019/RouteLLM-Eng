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
| 2026-09-17 | [sw_ranking 本地化与 Elo 求解器性能优化](2026-09-17-sw-ranking-localization.md) | ✅ 通过 | 本地 bge-m3 替代 OpenAI Embedding（55361 条 / 112s / $0）；**路由延迟 90% 在 `LogisticRegression.fit`**，换 newton-cholesky 后端到端 **394ms → 185ms（2.1×）**，模型排序完全一致 |
| 2026-09-17 | [镜像重建与容器切换](2026-09-17-image-rebuild-container-switch.md) | ✅ 通过 | 新代码上线无回归（镜像 675→704MB，仍未装 torch）；**关键发现：真实网关中路由开销占比 <5%，下游 LLM 生成占 1.7~16s** |
| 2026-09-17 | [sw_ranking 服务化上线](2026-09-17-sw-ranking-service-deployment.md) | ⚠️ 部分 | 链路打通：6071 host 服务 + 容器启用 `remote_sw_ranking`（零挂载保持轻量）；**但发现 win_rate 全挤在 0.689~0.693，0.5 阈值下永远走强模型**（bert 对比有正常区分度 0.297~0.453）→ 待查 |
| 2026-09-17 | [sw_ranking 区分度不足的根因诊断](2026-09-17-sw-ranking-discrimination-diagnosis.md) | ✅ 根因定位 | **`get_weightings` 动态范围仅 7 倍（14~100）→ 5.5 万条近似等权 → 全量参与时 elo_diff 全距为 0**；排除 sigmoid 饱和与 bge-m3 因素；候选修复（拉大动态范围）已验证方向正确，待 benchmark 论证 |

## 结论摘要（供快速引用）

### BERT 路由器（`routellm/bert_gpt4_augmented`）

- 实际架构 **xlm-roberta**，3 标签（上游命名为 `bert`）
- 加载 1.2s，单条推理 6.1ms（RTX 4090）
- `win_rate = 1 - sum(softmax(logits)[-2:])`，语义为"应路由到强模型的程度"
- **57 学科 MMLU 上 corr = -0.7123** —— 弱模型越不擅长的学科，win rate 越高
- 适用边界：依赖"query 文本能反映难度"。GSM8K 这类同质化数据集上无区分度

### sw_ranking 路由延迟构成（55361 条 arena，2026-09-17 实测）

```
encode_prompt (本地 bge-m3)      26.9ms    6.8%
similarity dot (55k x 1024)       6.3ms    1.6%
compute_elo_mle_with_tie        355.5ms   90.1%   ← 瓶颈
  ├─ X/Y 矩阵构造                34.4ms
  └─ LogisticRegression.fit     362.9ms
```

- 输入 `fit` 的形状：**110722 行 × 10 列**（样本极多、特征极少）
- 默认 `lbfgs` 每轮全量扫 11 万行，与数据形状不匹配
- 换 **`newton-cholesky`**：Elo 求解 331ms → 87ms（3.8×），端到端 **394ms → 185ms**（2.1×）
- 数值：elo 最大偏差 0.72 分，模型排序完全一致，winrate 偏移 < 0.005
- 传 `solver="lbfgs"` 可复现旧结果

> 求解器性能对 `sample_weight` 的**逐元素排列**敏感。合成权重（uniform / beta）下 newton-cholesky 反而更慢，性能测试必须走真实路由链路取 `get_weightings(cosine_sims)`。

### ⚠️ 已知问题：sw_ranking 区分度不足（2026-09-17，根因已定位）

| prompt | sw_ranking | bert |
|---|---|---|
| hi | 0.6892 | 0.4007 |
| What is 1+1? | 0.6931 | 0.2970 |
| 证明√2无理数 | 0.6920 | 0.4533 |

分布实测（n=300）：sw_ranking **std=0.0008 / 全距 0.0043**，bert **std=0.15 / 全距 0.844**（相差约 190 倍）。

**根因**：`get_weightings(sims) = 10 * 10^(sim/max_sim)` 的**动态范围仅 7 倍**（实测 14.3~100）。
最不相似的 battle 也拿到约 1/7 权重 → 5.5 万条样本**近似等权**参与 LogisticRegression
→ 单 prompt 的相似度差异被平均掉。

**决定性证据**（top-K 对照）：

| topK | 两个极端 prompt 的 elo_diff | 全距 |
|---|---|---|
| 100 | [2.17, 89.23] | 87.05 |
| 全部 55361 | **[141.013, 141.013]** | **0** |

**已排除的假设**：sigmoid 饱和（斜率仅降 15%，非 150 倍）；elo_strong/weak 同增同减（corr=0.07）；bge-m3 编码能力（向量间相似度 0.3~0.7 正常）。

**候选修复**（已验证方向，未决策）：拉大权重动态范围。`pow p=16` → winrate 全距 0.4002；
`top 1%` → 0.1764。`pow p=8` 的逐条结果方向正确（hi→0.654 最低，证明√2→0.698 最高）。

**建议**：生产路由继续用 `remote_bert`；修复前需先用官方 thresholds 数据集对标 +
benchmark 准确率论证，不可擅自改动上游算法语义。

### 环境版本（43 号机 rag-dev 环境）

```
Python 3.11.13 / torch 2.9.0+cu128 / sentence-transformers 4.1.0 / datasets 4.4.1 / pytest 8.4.1
```

### 环境版本（33 号机 venv）

```
Python 3.10.12 / torch 2.14.0 / transformers 5.17.0 / datasets 5.0.1 / litellm 1.101.0
```

`import` 全部通过，`transformers 5.x` 未破坏导入。
