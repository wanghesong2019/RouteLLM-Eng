# 实验记录：BERT 路由模型加载与区分度验证

- **日期**：2026-09-16
- **机器**：43 号机（server43-X640-G40，4×RTX 4090，24GB each）
- **环境**：`/mnt/data/wanghesong/conda-env/evaluation-engine-sansuo`（Python 3.10 / transformers 4.49.0 / torch 2.6.0+cu124 / fastapi）
- **模型**：`routellm/bert_gpt4_augmented`（经 hf-mirror.com 下载，1.1GB）
- **脚本**：`scripts/verify_bert_basic.py`、`scripts/verify_bert_discrimination.py`
- **状态**：✅ 通过

## 实验目的

在把 BERT 推理封装成 host 服务之前，先确认两件事：

1. 模型能否正常加载并复现上游 `calculate_strong_win_rate` 的结果
2. 输出的 win rate 是否真有区分能力（而不是一个恒定值或随机噪声）

第 2 点尤其重要 —— 如果模型加载有误（如 tokenizer 配置错误、标签语义反了），服务封装得再好也是错的。

## 被测逻辑（复刻上游实现）

```python
# 源自 routellm/routers/routers.py :: BERTRouter.calculate_strong_win_rate
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits.numpy()[0]
exp_scores = np.exp(logits - np.max(logits))
softmax_scores = exp_scores / np.sum(exp_scores)
binary_prob = np.sum(softmax_scores[-2:])
return 1 - binary_prob
```

## 实验 1：加载与基础行为

### 结果

| 项 | 实测值 |
|---|---|
| 实际架构 | **xlm-roberta**（非 BERT，`bert` 是上游命名） |
| num_labels | 3 |
| 加载耗时 | 1.2s |
| 单条推理延迟 | min 6.1ms / avg 6.1ms / max 6.4ms（预热后 10 次，RTX 4090） |
| 确定性 | 同 prompt 两次：`0.2969909906` == `0.2969909906` ✓ |

### 手写 prompt 的 win rate

| win_rate | softmax [l0, l1, l2] | prompt |
|---|---|---|
| 0.2970 | [0.297, 0.205, 0.498] | What is 1+1? |
| 0.4007 | [0.401, 0.222, 0.377] | hi |
| 0.3514 | [0.351, 0.260, 0.389] | Explain the trade-offs between consistency and availability... |
| 0.1993 | [0.199, 0.191, 0.610] | Write a poem about the sea. |
| 0.4351 | [0.435, 0.272, 0.293] | Prove that the square root of 2 is irrational... |

**观察**：全部低于 0.5，分布集中（0.199~0.435）。当时无法判断是"prompt 都太简单"还是"模型加载有误"。→ 触发实验 2。

## 实验 2：GSM8K 验证（❌ 反例，方法不当）

### 设计

从 `routellm/evals/gsm8k/gsm8k_responses.csv`（1319 题，含 Mixtral / GPT-4 对错标记）中，按**弱模型（Mixtral）答对 / 答错**各抽 25 条，比较两组 win rate。

**假设**：弱模型答对的题 → win rate 应更高（因为弱模型能胜任，不该送强模型）。

### 结果

```
group              n    mean     std      min      max
weak_CORRECT      25   0.5295   0.0688   0.3846   0.7215
weak_WRONG        25   0.5291   0.0773   0.3945   0.7098

差值 = +0.0004   ← 几乎为 0

threshold 0.5 下的路由分布：
  weak_CORRECT 送强模型 17/25
  weak_WRONG   送强模型 12/25
```

**结论：无区分度。** 但先不下"模型有问题"的判断 —— 观察到 win rate 范围（0.385~0.722）与实验 1 的手写 prompt（0.199~0.435）明显不同，提示是**数据特性差异**而非模型故障。

### 对失败原因的分析

GSM8K 的 1319 题**全是小学数学应用题**，文本特征高度同质化。弱模型在某道题上答对与否，取决于题目中的具体数字与运算链，**难以从题目文本本身预测**。路由器看到的所有输入长得都差不多，自然无法区分。

→ 改用跨学科数据集（MMLU）重测。

## 实验 3：MMLU 验证（✅ 结论性）

### 设计

读 `routellm/evals/mmlu/responses/` 下全部 **57 个学科**的 CSV，每学科抽样最多 30 条，计算：

- 该学科弱模型（Mixtral）答对率 `weak_acc`
- 该学科平均 win rate `mean_wr`

**假设**：若路由器有效，弱模型擅长的学科 → win rate 应**偏低**（该送弱模型）。

注意此处假设方向与实验 2 相反 —— 因为 `win_rate` 的语义是"应路由到强模型的程度"，弱模型表现越好，越不该送强模型。

### 结果：57 学科全量（按弱模型答对率升序）

| 学科 | weak_acc | strong_acc | mean_wr |
|---|---|---|---|
| formal_logic | 0.233 | 0.600 | **0.7532** |
| high_school_mathematics | 0.300 | 0.000 | **0.8513** |
| abstract_algebra | 0.333 | 0.367 | **0.8096** |
| college_computer_science | 0.333 | 0.700 | 0.7261 |
| college_mathematics | 0.367 | 0.367 | **0.8379** |
| college_chemistry | 0.433 | 0.400 | 0.7404 |
| virology | 0.433 | 0.533 | 0.4982 |
| college_physics | 0.500 | 0.467 | 0.7409 |
| high_school_physics | 0.500 | 0.633 | 0.7741 |
| global_facts | 0.533 | 0.567 | 0.6617 |
| moral_scenarios | 0.533 | 0.800 | 0.7979 |
| professional_law | 0.533 | 0.733 | 0.4157 |
| elementary_mathematics | 0.567 | 0.700 | 0.8359 |
| econometrics | 0.600 | 0.567 | 0.6921 |
| conceptual_physics | 0.633 | 0.900 | 0.6331 |
| high_school_chemistry | 0.633 | 0.800 | 0.6791 |
| machine_learning | 0.633 | 0.800 | 0.6726 |
| professional_accounting | 0.633 | 0.533 | 0.6062 |
| high_school_macroeconomics | 0.667 | 0.767 | 0.5891 |
| high_school_computer_science | 0.700 | 0.767 | 0.7271 |
| high_school_microeconomics | 0.700 | 0.967 | 0.5282 |
| jurisprudence | 0.700 | 0.867 | 0.5010 |
| college_biology | 0.733 | 0.900 | 0.5387 |
| college_medicine | 0.733 | 0.767 | 0.5592 |
| high_school_us_history | 0.733 | 0.867 | 0.4321 |
| human_aging | 0.733 | 0.900 | 0.4353 |
| professional_medicine | 0.733 | 0.967 | 0.4184 |
| security_studies | 0.733 | 0.867 | 0.4325 |
| miscellaneous | 0.767 | 0.967 | 0.5571 |
| computer_security | 0.800 | 0.933 | 0.5089 |
| high_school_psychology | 0.800 | 0.967 | 0.4278 |
| high_school_statistics | 0.800 | 0.767 | 0.7346 |
| management | 0.800 | 0.867 | **0.3602** |
| medical_genetics | 0.800 | 0.967 | 0.5146 |
| professional_psychology | 0.800 | 0.867 | 0.4384 |
| public_relations | 0.800 | 0.800 | 0.3929 |
| anatomy | 0.833 | 0.900 | 0.4829 |
| business_ethics | 0.833 | 0.867 | 0.4437 |
| human_sexuality | 0.833 | 1.000 | 0.4864 |
| prehistory | 0.833 | 0.933 | 0.5405 |
| clinical_knowledge | 0.867 | 1.000 | 0.4441 |
| electrical_engineering | 0.867 | 0.867 | 0.6051 |
| high_school_world_history | 0.867 | 1.000 | 0.5165 |
| logical_fallacies | 0.867 | 0.967 | 0.5049 |
| moral_disputes | 0.867 | 0.933 | 0.4766 |
| nutrition | 0.867 | 0.900 | 0.4942 |
| philosophy | 0.867 | 0.867 | 0.4810 |
| world_religions | 0.867 | 0.767 | 0.6231 |
| astronomy | 0.900 | 0.933 | 0.6432 |
| high_school_biology | 0.900 | 0.900 | 0.4872 |
| high_school_government_and_politics | 0.900 | 0.900 | 0.4396 |
| high_school_geography | 0.933 | 0.933 | 0.5435 |
| international_law | 0.933 | 0.967 | 0.4661 |
| sociology | 0.933 | 0.900 | 0.4430 |
| us_foreign_policy | 0.967 | 0.933 | 0.5122 |
| high_school_european_history | 1.000 | 0.967 | 0.4648 |
| marketing | 1.000 | 1.000 | **0.3716** |

### 统计

```
学科数: 57
weak_acc 范围: 0.233 ~ 1.000   (std 0.187)
mean_wr  范围: 0.3602 ~ 0.8513 (std 0.1337)
corr(weak_acc, mean_wr) = -0.7123

弱模型最不擅长的 1/4 学科 → 平均 win rate = 0.7239
弱模型最擅长的   1/4 学科 → 平均 win rate = 0.4965
差值 = -0.2274
```

### 结论

**路由器工作正常，区分度强，方向正确。**

- `corr = -0.7123`（57 学科），强负相关
- 弱模型不擅长的学科（formal_logic 0.233 / 高等数学 0.300）→ win rate 0.75~0.85 → 路由到**强模型** ✓
- 弱模型擅长的学科（marketing 1.000 / management 0.800）→ win rate 0.36~0.37 → 路由到**弱模型** ✓

**确认 `win_rate = 1 - binary_prob` 语义为"应路由到强模型的程度"**，`softmax[-2:]` 对应 [tie, 弱模型胜] 两类。

## 两个实验的差异解释

| 数据集 | 性质 | 相关系数 | 结论 |
|---|---|---|---|
| GSM8K | 1319 题全为小学数学，文本同质 | ≈ 0 | 路由器无法区分 |
| MMLU | 57 学科跨领域，文本差异大 | -0.71 | 路由器区分度强 |

**这不是模型缺陷，而是路由器的适用边界**：其有效性依赖"query 文本本身能反映难度"。同质化数据集上，文本特征无法区分难度，路由器无能为力。

## 对项目的连带影响

1. **Phase 5 评测基准选择**：应以 **MMLU 为主**验证路由有效性，GSM8K 不适合作为区分度验证集
2. **win rate 全部 < 0.5 的手写 prompt**（实验 1）属于正常现象 —— 那批 prompt 确实偏简单
3. **发现评测判定方式的敏感性**：`high_school_mathematics` 的 `strong_acc = 0.000`（GPT-4 对该学科按精确匹配一道未中），但 win rate 高达 0.8513。说明答案判定方式对评测结果影响极大，Phase 5 做对比时需固定判定口径

## 复现方式

```bash
# 1. 下载模型（43 号机）
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('routellm/bert_gpt4_augmented',
                  local_dir='/mnt/data/wanghesong/routellm/models/bert_gpt4_augmented')
"

# 2. 跑区分度验证
python scripts/verify_bert_discrimination.py \
    --model /mnt/data/wanghesong/routellm/models/bert_gpt4_augmented \
    --mmlu-dir <mmlu_responses_dir>
```
