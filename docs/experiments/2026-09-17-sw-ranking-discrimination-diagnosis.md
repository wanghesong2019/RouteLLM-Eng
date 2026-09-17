# 实验记录：sw_ranking 区分度不足的根因诊断

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **环境**：`/mnt/data/wanghesong/conda-env/rag-dev`
- **数据**：`lmsys-arena-human-preference-55k`（preprocess 后 55361 条）
- **脚本**：`scripts/diagnose_sw_ranking_distribution.py`、`trace_sw_ranking_pipeline.py`、`probe_elo_weight_sensitivity.py`、`probe_weight_mapping.py`
- **状态**：✅ 根因已定位，⚠️ 修复方案待论证（未改代码）

## 问题

sw_ranking 服务化上线后（见 2026-09-17-sw-ranking-service-deployment.md）发现：
win_rate 全部挤在 0.689~0.693，用 threshold=0.5 会导致**所有请求都路由到强模型**。

## 实验 1：分布诊断（300 条真实 arena prompt）

| 统计量 | sw_ranking | bert（对照） |
|---|---|---|
| min | 0.689927 | 0.0538 |
| p25 | — | 0.3165 |
| **p50** | **0.691919** | 0.4052 |
| p75 | 0.6924 | 0.4807 |
| p95 | 0.6932 | 0.6612 |
| max | **0.694207** | 0.8980 |
| **std** | **0.000797** | **0.150** |
| **全距** | **0.004280** | **0.844** |

**sw_ranking 的 std 比 bert 小约 190 倍。**

注：sw_ranking 确实在计算（300 条中有 289 个不同值，非常量），但变化幅度无实用意义。

### 阈值标定（官方方法）

查上游 `calibrate_threshold.py`：

```python
threshold = win_rate 分布的 quantile(q = 1 - strong_model_pct)
```

即按「期望强模型占比」反推阈值，而非固定 0.5。实测：

| 目标强模型占比 | sw_ranking 阈值 | bert 阈值 |
|---|---|---|
| 50% | 0.69195 | 0.40515 |
| 30% | 0.69230 | 0.46493 |
| 20% | 0.69253 | 0.50534 |
| 10% | 0.69293 | 0.59105 |

sw_ranking 从 20% 调到 10% 占比，阈值只差 **0.0004** —— 浮点噪声量级，实际不可用。

**相关性**：`corr(sw_ranking, bert) = 0.076` —— 两个路由的判定几乎无关。

## 实验 2：逐环节追踪

用 7 个语义差异极大的 prompt 追踪链路中间量：

```
prompt → bge-m3 向量 → cosine 相似度 → get_weightings → 加权 Elo 回归 → winrate
```

| 环节 | 全距 | std |
|---|---|---|
| B: `sim_max` | 0.4443 | 0.1411 |
| B: `sim_mean` | 0.1071 | 0.0335 |
| C: `w_min` | 4.3036 | 1.6431 |
| C: `w_mean` | 17.0755 | 5.6925 |
| D: `elo_strong` | 2.8914 | 0.9206 |
| D: `elo_weak` | 0.8702 | 0.2715 |
| **E: winrate** | **0.003842** | **0.001157** |

**关键**：`elo_strong - elo_weak` 的差值全距仅 **3.13 分**（138.36 ~ 141.48）。

### 排除的假设

1. **sigmoid 饱和** —— 已排除。实测 `diff=140` 处斜率 0.001229 vs `diff=0` 处 0.001439，
   灵敏度只降 15%，**不是 150 倍**（初稿曾误写，已更正）。
   真实原因是 diff 本身波动小（3.13 分），而非 sigmoid 饱和。

2. **elo_strong 与 elo_weak 同增同减** —— 已排除。
   `corr(elo_strong, elo_weak) = 0.070`，几乎不相关。

## 实验 3：权重敏感度探测

### 3.1 尺度不变性确认

| 权重缩放系数 | elo_diff |
|---|---|
| 0.01 | 138.357822 |
| 0.1 | 138.357822 |
| 1.0 | 138.357822 |
| 10.0 | 138.357822 |
| 100.0 | 138.357821 |

**权重整体缩放 10000 倍，结果小数点后 6 位不变。** Elo 回归对权重尺度完全不敏感。

### 3.2 top-K 对照（关键证据）

| topK | 两个极端 prompt 的 elo_diff | 全距 | winrate 全距 |
|---|---|---|---|
| 10 | [4963.5, …] | — | 1.0000 |
| 100 | [2.17, 89.23] | 87.05 | **0.1225** |
| 1000 | [14.34, 98.34] | 84.00 | 0.1172 |
| 10000 | [118.87, 136.34] | 17.46 | 0.0220 |
| **55361（全部）** | **[141.01, 141.01]** | **0.0000** | **0.0000** |

**用全部 55361 条时，语义天差地别的两个 prompt（"hi" vs "证明√2无理数"）
得到完全相同的 elo 差值（141.013）。** 区分度被完全抹平。

## 根因

`get_weightings(sims) = 10 * 10^(sim / max_sim)` 的**动态范围仅约 7 倍**
（实测区间 14.3 ~ 100）。

后果：最不相似的 battle 也拿到 14.3 分（约最大权重的 1/7），
5.5 万条样本**近似等权**参与 LogisticRegression 拟合
→ 单个 prompt 的相似度差异被 5 万余条样本平均掉
→ 不同 prompt 的回归结果趋同 → winrate 无区分度。

**这不是 bge-m3 引入的问题**（bge-m3 能正常区分 prompt：向量间相似度 0.3~0.7），
也**不是本地化改造引入的** —— 它是 `get_weightings` 映射函数的固有特性。

> 上游是否同样如此：未验证。上游用 `text-embedding-3-small`（1536 维）+
> `gpt4_judge_battles` 数据集，相似度分布可能不同。需官方
> `routellm/lmsys-arena-human-preference-55k-thresholds` 数据集对标，
> 本机网络受限（huggingface.co 仅 IPv6 不可达），待手动获取。

## 候选修复（已验证有效性，未决策）

把权重动态范围拉大，区分度即可恢复，且**方向正确**：

| 权重映射 | 权重区间 | winrate 全距 | 评价 |
|---|---|---|---|
| 上游 `w_orig` | 14.3~100 | 0.0038 | ❌ 无 |
| `pow p=4` | 1~54.6 | 0.0111 | ⚠️ 弱 |
| `pow p=8` | 1~2.98e3 | 0.0441 | ⚠️ 弱 |
| **`pow p=16`** | 1~8.89e6 | **0.4002** | ✅ 好 |
| `exp p=10` | 4.5e-5~1 | 0.0932 | ✅ 好 |
| `top 10%` | 0~1 | 0.0628 | ✅ 好 |
| `top 5%` | 0~1 | 0.1216 | ✅ 好 |
| **`top 1%`** | 0~1 | **0.1764** | ✅ 好 |

### `pow p=8` 的逐条结果（方向验证）

| prompt | elo_diff | winrate |
|---|---|---|
| hi | 110.738 | **0.6542** ← 最低（简单题 → 倾向弱模型）✓ |
| Write a haiku about autumn leaves. | 124.529 | 0.6719 |
| Explain the trade-offs between consistency and availability… | 143.652 | 0.6957 |
| Prove that the square root of 2 is irrational… | 145.738 | **0.6982** ← 最高（难题 → 倾向强模型）✓ |

**方向完全符合直觉。**

### 修复方案的风险

1. **改动范围**：`get_weightings` 是 `SWRankingRouter` 与 host 服务的共用逻辑，
   两处都需同步（host 服务是刻意的代码重复）。
2. **语义变化**：这实质上是**改变路由算法**，不再是上游的 sw_ranking。
   需在文档中明确标注差异，避免与官方效果对比时产生误解。
3. **参数选择缺乏依据**：`p=16` / `p=8` / `top 1%` 哪个最优，目前只看了
   5~7 个 prompt 的"方向正确性"，**未用 benchmark 做准确率验证**。
4. **可能与官方设计意图冲突**：上游选择 7 倍动态范围或许有其理由
   （如避免过拟合到少数相似样本）。需谨慎。

### 建议的论证路径

1. 先用**官方 thresholds 数据集**对标：上游 sw_ranking 的 win_rate 分布
   是否同样窄？若同样窄，说明这是上游的已知特性，不应擅自改。
2. 若上游分布正常，则用 **MMLU / GSM8K benchmark** 评测不同映射的
   路由准确率（`routellm/evals/` 有预计算数据，无需真实 LLM API），
   选准确率最高者。
3. 参数选定后按 TDD 补测试，同步改 host 服务与容器侧。

## 当前部署建议

**生产路由继续使用 `remote_bert`。** `remote_sw_ranking` 已注册并可用，
但在阈值/映射问题解决前不应作为默认路由。

## 复现方式

```bash
export PY=/mnt/data/wanghesong/conda-env/rag-dev/bin/python
export ROOT=/mnt/data/wanghesong/routellm
export PYTHONPATH=$ROOT/RouteLLM-Eng
export CUDA_VISIBLE_DEVICES=1
unset OPENAI_API_KEY

# 1. 分布诊断（实时计算 300 条，约 1 分钟）
$PY scripts/diagnose_sw_ranking_distribution.py \
    --battles-csv $ROOT/arena_train.csv \
    --embeddings $ROOT/embeddings/arena_embeddings.npy \
    --model-path $ROOT/models/bge-m3/models/BAAI--bge-m3/snapshots/master \
    --sample-size 300

# 2. 逐环节追踪
$PY scripts/trace_sw_ranking_pipeline.py

# 3. 权重敏感度
$PY scripts/probe_elo_weight_sensitivity.py

# 4. 候选映射对比
$PY scripts/probe_weight_mapping.py
```
