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
| 2026-09-17 | [sw_ranking 区分度不足的根因诊断](2026-09-17-sw-ranking-discrimination-diagnosis.md) | ✅ 已修复 | **根因：只接了 arena 未接 gpt4_judge_battles 数据集**（官方需拼接两个）；补齐后 mean 0.2148 vs 官方 0.2166（**均值比 0.9914**），逐条相关 **0.8052**（修复前 0.076） |
| 2026-09-17 | [sw_ranking 阈值标定与路由选型建议](2026-09-17-sw-ranking-threshold-calibration.md) | ✅ 决策 | 阈值须用 quantile 标定（官方方法），50% 占比 → 0.2165；**逐 prompt 对比发现 sw_ranking 是「配额分配」而非「难度判断」**（跨度 0.007 vs bert 0.30）→ **生产默认用 `remote_bert`** |
| 2026-09-17 | [bert 路由的论文指标评测（APGR）](2026-09-17-bert-apgr-evaluation.md) | ✅ 通过 | **MMLU APGR 0.5267 / GSM8K 0.5333**（对照全弱 0.68/0.64、全强 0.79/0.86）；评测无需真实 LLM API，5700 题评分 7.9s |

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

### ✅ 已修复：sw_ranking 曾与官方有 3.2 倍系统偏差（2026-09-17）

**根因：只接入了官方两个数据集中的一个。**

```
官方:   arena (lmsys/...-55k)  +  judge (routellm/gpt4_judge_battles)
修复前: 仅 arena
```

**修复效果**（与官方 thresholds 数据集逐条对标，抽样 100 条）：

| 指标 | 官方 | 修复前 | 修复后 |
|---|---|---|---|
| mean | 0.2166 | 0.6919 | **0.2148（均值比 0.9914）** |
| std | 0.0025 | 0.0008 | **0.0030** |
| 逐条相关 | — | 0.076 | **0.8052** |

**关键佐证**：补齐 judge 后 Elo 估计被显著改变，连强弱关系都翻转：
```
修复前: strong(gpt-4-1106-preview)=1154.6, weak(mixtral)=1015.3, 差 +138.4
修复后: strong=1109.8, weak=1335.7, 差 -226.0
```

**官方 sw_ranking 分布参考**（n=57477）：mean 0.2166，std 0.0025，全距 0.0210 ——
分布本身极窄，是算法固有特性，**必须用 quantile 标定阈值**
（`threshold = quantile(1 - strong_pct)`），不能沿用 bert 的 0.4~0.6 量级。

| 目标强模型占比 | 官方阈值 | 本实现阈值 |
|---|---|---|
| 50% | 0.216473 | 0.214868 |
| 20% | 0.218800 | 0.217361 |
| 10% | 0.219952 | 0.219144 |

**数据获取方式**（两者均可经 hf-mirror 直接 curl，非 gated）：
- thresholds: `routellm/lmsys-arena-human-preference-55k-thresholds`（2.1MB）
- judge battles: `routellm/gpt4_judge_battles`（159MB，parquet）

### 路由选型结论（2026-09-17 决策）

**生产默认用 `remote_bert`，`remote_sw_ranking` 保留为可选。**

| | bert | sw_ranking |
|---|---|---|
| 机制 | 分类模型判别 prompt 难度 | 相似度加权 Elo 回归 |
| 输出语义 | **单请求难度**评分 | **配额分配**（按分位数切分） |
| 分布跨度 | 0.30+（0.172~0.472） | <0.01（0.2099~0.2167） |
| 阈值鲁棒性 | 0.5 有明确含义，容错大 | 容错仅 0.0035，部署风险高 |
| 难度感知 | ✅ 正确（难题→高值） | ❌ 排序无意义 |

sw_ranking 的 win_rate 阈值须用官方 quantile 方法标定
（`threshold = quantile(1 - strong_pct)`），实测：

| 目标强模型占比 | 阈值 |
|---|---|
| 50% | 0.2165 |
| 20% | 0.2188 |
| 10% | 0.2200 |

**注意**：sw_ranking 阈值绝不可沿用 bert 的 0.4~0.6 量级 ——
分布位置完全不同（这曾是本地化初期"永远走强模型"的原因）。

### 环境版本（43 号机 rag-dev 环境）

```
Python 3.11.13 / torch 2.9.0+cu128 / sentence-transformers 4.1.0 / datasets 4.4.1 / pytest 8.4.1
```

### 环境版本（33 号机 venv）

```
Python 3.10.12 / torch 2.14.0 / transformers 5.17.0 / datasets 5.0.1 / litellm 1.101.0
```

`import` 全部通过，`transformers 5.x` 未破坏导入。
