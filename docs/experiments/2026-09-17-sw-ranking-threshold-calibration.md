# 实验记录：sw_ranking 阈值标定与路由选型建议

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **状态**：✅ 完成标定，给出选型建议
- **相关**：[区分度诊断](2026-09-17-sw-ranking-discrimination-diagnosis.md)、[服务化上线](2026-09-17-sw-ranking-service-deployment.md)

## 背景

补齐 `gpt4_judge_battles` 后，sw_ranking 的 win_rate 已与官方对齐
（均值比 0.9914、逐条相关 0.8052）。但分布本身极窄（std≈0.003），
需要按官方方法标定阈值，并明确它与 bert 的适用边界。

## 实验 1：阈值标定（官方 quantile 方法）

上游 `calibrate_threshold.py` 的做法：

```python
threshold = win_rate 分布的 quantile(q = 1 - strong_model_pct)
```

即按「期望强模型调用占比」反推，而非固定阈值。

在**官方 thresholds 数据**上标定（n=57477）：

| 目标强模型占比 | 阈值 | 含义 |
|---|---|---|
| 80% | 0.2143 | 多数请求走强模型（贵、质量高） |
| 70% | 0.2151 | |
| **50%** | **0.2165** | 平衡（官方默认基准） |
| 30% | 0.2180 | |
| 20% | 0.2188 | 偏省 |
| 10% | 0.2200 | 最省 |

官方分布参考：min 0.2090 / p25 0.2147 / p50 0.2165 / p75 0.2183 / max 0.2300

**注意阈值的敏感度**：从 50% 占比到 10% 占比，阈值仅从 0.2165 变到 0.2200
（差 0.0035）。**这意味着部署时阈值必须高精度设定**，且不可沿用 bert 的
0.4~0.6 量级 —— 否则行为完全错误（这正是本地化初期"永远走强模型"的原因）。

## 实验 2：逐 prompt 对比（关键发现）

同一批语义难度差异极大的 prompt，两个路由的 win_rate：

| prompt | sw_ranking | bert |
|---|---|---|
| hi | 0.21346 | 0.40069 |
| What is 1+1? | 0.21655 | 0.29699 |
| What is the capital of France? | 0.20989 | 0.36457 |
| Write a haiku about autumn leaves. | 0.21404 | 0.20184 |
| Explain the trade-offs between consistency and availability… | 0.21669 | **0.17183** |
| Prove that the square root of 2 is irrational… | 0.21584 | **0.47156** |

### bert 的行为（符合预期）

- 范围 0.172 ~ 0.472，跨度 0.30
- **难度感知正确**：一致性权衡（最难）→ 0.172 最低；证明√2 → 0.472 最高
- 简单题（hi 0.401、1+1 0.297）落在低位

### sw_ranking 的行为（固有特性）

- 范围 0.2099 ~ 0.2167，跨度仅 **0.0068**
- **排序基本无意义**：
  - 最简单的 "hi"（0.2135）**高于** "法国首都"（0.2099）
  - 最难的 "证明√2"（0.2158）**低于** "1+1"（0.2166）

### 结论：两者语义不同

| | bert | sw_ranking |
|---|---|---|
| 机制 | 分类模型直接判别 prompt 难度 | 相似度加权 Elo 回归 |
| 输出语义 | **单请求难度**评分 | **配额分配**（按分位数切分） |
| 分布跨度 | 0.30+ | <0.01 |
| 适用 | 按难度逐请求路由 | 按目标比例分配强模型调用量 |

**这不是 sw_ranking 的缺陷** —— 官方发布数据的分布同样极窄
（std=0.0025，见 2026-09-17-sw-ranking-discrimination-diagnosis.md）。
它的设计意图是「在给定成本预算下，把强模型调用分配给相对更需要的一批请求」，
而非逐请求判断难度。

## 实验 3：容器端到端验证

容器已配双路由，调用方通过 `model` 字段选择：

```
router-remote_bert-0.5          ← 推荐默认
router-remote_sw_ranking-0.5    ← 可选（需配 0.2165 量级阈值）
```

实测（阈值 0.215 / 0.217 两档）：

```
                     th=0.215           th=0.217
hi                → Flash(弱)         → Flash(弱)
What is 1+1?      → Pro(强)           → Flash(弱)
证明√2无理数        → Pro(强)           → Flash(弱)
```

阈值 0.217 下三个 prompt 全部走弱模型 —— 印证了分布极窄导致的**阈值高敏感性**。

## 选型建议（采纳）

**生产默认使用 `remote_bert`。**

理由：

1. **语义匹配**：bert 输出的是单请求难度评分，符合"难问题送强模型"的直觉，
   且 MMLU 上已验证区分度（corr = -0.7123，见 2026-09-16-bert-router-validation.md）
2. **阈值鲁棒**：bert 分布跨度 0.30+，阈值 0.5 有明确含义且容错空间大；
   sw_ranking 的阈值容错仅 0.0035，部署风险高
3. **sw_ranking 的定位**：适合"按预算配额分配"场景，需要配套的监控与动态标定，
   当前阶段不具备

**sw_ranking 保留为可选路由**，用于：
- 与 bert 做路由策略对比实验（评测阶段）
- 需要严格控制强模型调用比例的预算敏感场景

## 未完成事项

1. host 服务（6070/6071）为 nohup 裸进程，未纳入 systemd —— 机器重启不自恢复
2. 容器 `--restart no`
3. sw_ranking 若要在生产使用，需先建立**动态阈值标定机制**
   （定期用真实流量重新计算分位数）

## 复现方式

```bash
export PY=/mnt/data/wanghesong/conda-env/rag-dev/bin/python
export ROOT=/mnt/data/wanghesong/routellm

# 阈值标定（基于官方数据）
$PY -c "
import numpy as np, pandas as pd
v = pd.read_parquet('$ROOT/official_thresholds.parquet')['sw_ranking'].values
for pct in [0.8,0.7,0.5,0.3,0.2,0.1]:
    print(f'{pct*100:3.0f}% -> {np.quantile(v, 1-pct):.4f}')
"

# 逐 prompt 对比（服务需已启动）
for p in 'hi' 'What is 1+1?' 'Prove that the square root of 2 is irrational.'; do
  echo "sw_ranking: $(curl -s -X POST localhost:6071/v1/score \
    -H 'Content-Type: application/json' \
    -d "{\"prompts\":[\"$p\"]}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["results"][0]["win_rate"])')"
  echo "bert      : $(curl -s -X POST localhost:6070/v1/score \
    -H 'Content-Type: application/json' \
    -d "{\"prompts\":[\"$p\"]}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["results"][0]["win_rate"])')"
done
```
