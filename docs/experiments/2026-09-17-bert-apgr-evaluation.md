# 实验记录：bert 路由的论文指标评测（APGR）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **路由**：`remote_bert`（`routellm/bert_gpt4_augmented`，经 host 服务 6070）
- **脚本**：`scripts/eval_router_apgr.py`
- **原始结果**：`results/apgr_bert.json`
- **状态**：✅ 通过

## 指标定义（论文标准，源自 routellm/evals/evaluate.py）

```
APGR = (router_AUC - weak_AUC) / (strong_AUC - weak_AUC)
```

- **0** = 与「全部请求走弱模型」等效
- **1** = 达到理论最优（先知路由，知道哪题该走强模型）
- 物理含义：**在成本-质量权衡曲线上，路由器捕获了强弱模型性能差距的多少比例**

评测无需真实 LLM API —— benchmark 数据自带弱/强模型的**预计算对错标记**，
路由器只需产出 win_rate 即可扫出成本-质量曲线。

## 实测结果

| 数据集 | 题量 | APGR | 全弱模型 | 全强模型 | 路由器 AUC |
|---|---|---|---|---|---|
| **MMLU**（57 学科） | 5700 | **0.5267** | 0.6832 | 0.7947 | 0.7419 |
| **GSM8K** | 1319 | **0.5333** | 0.6384 | 0.8567 | 0.7548 |

### 结果解读

以 GSM8K 为例：

- 全走弱模型 → 准确率 63.8%
- 全走强模型 → 准确率 85.7%
- 路由器 → 约 75%（在强模型调用占比 50% 附近）
- **即：用约一半的强模型调用量，达到全强模型约 3/4 的效果**

MMLU 同理：全弱 68.3%、全强 79.5%，路由器捕获其中 52.7% 的差距。

## 性能

| 项 | 值 |
|---|---|
| MMLU 5700 题评分 | 7.9s |
| GSM8K 1319 题评分 | 1.2s |
| BERT 单条推理 | 6.1ms（RTX 4090，见 2026-09-16-bert-router-validation.md） |

评测全程无需外部 LLM API，可离线重跑。

## 在简历/面试中的定位（重要）

**APGR 0.5267 / 0.5333 是 bert 模型本身的能力，不是改造的成果。**
面试表述应准确定位，例如：

> 接入并工程化 RouteLLM 路由网关，在 MMLU / GSM8K 上复现论文指标
> APGR 0.53（对照：全弱 0.64 / 全强 0.86）；
> 针对原实现路由延迟瓶颈（`LogisticRegression.fit` 占 90%）改用
> newton-cholesky 求解器，路由延迟降低 2.1 倍（394ms → 185ms）；
> 将 Embedding 推理下沉为本地服务，去除 OpenAI 依赖并保持 675MB 轻量容器。

改造侧的可量化成果（详见各实验记录）：

| 项 | 数据 |
|---|---|
| 路由延迟优化 | 394ms → 185ms（2.1×），瓶颈定位到 `LogisticRegression.fit`（占 90%） |
| 去除外部依赖 | 本地 bge-m3 替代 OpenAI Embedding，55361 条向量 112s / $0 |
| 轻量网关 | 容器 675→704MB（无 torch），重推理下沉 host |
| 修复系统偏差 | sw_ranking win_rate 与官方差 3.2 倍 → 补齐数据集后均值比 0.9914、相关 0.8052 |
| 测试体系 | 从零建立，56+ 测试，TDD 流程 |

## 复现方式

```bash
export PY=/mnt/data/wanghesong/conda-env/rag-dev/bin/python
cd /mnt/data/wanghesong/routellm/RouteLLM-Eng

# 前置：bert 推理服务须在 6070 运行
#   python services/inference_server.py --model-type bert \
#       --model-path <bert ckpt> --gpu 0 --port 6070

env PYTHONPATH=$PWD $PY scripts/eval_router_apgr.py \
    --bert-url http://127.0.0.1:6070 \
    --mmlu-dir routellm/evals/mmlu/responses \
    --gsm8k-csv routellm/evals/gsm8k/gsm8k_responses.csv \
    --sample-per-subject 100 \
    --out-dir results
```

参数说明：
- `--sample-per-subject 100`：每学科抽样 100 题（57×100=5700）；设 0 用全量
- 阈值扫描范围：0.00~1.00 步长 0.01（见脚本 `thresholds`）

## 未做的事（范围限制）

- 未评测 `sw_ranking` / `mf` / `causal_llm` 的 APGR（本次聚焦 bert）
- 未与论文报告的官方数字逐位对比（论文用的是官方标定阈值配置）
- MT-Bench 未评测（需 judge 分数，数据在 `routellm/evals/mt_bench/`）
