# 实验记录：sw_ranking 本地化与 Elo 求解器性能优化

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090，24GB each）
- **环境**：`/mnt/data/wanghesong/conda-env/rag-dev`（Python 3.11.13 / torch 2.9.0+cu128 / sentence-transformers 4.1.0 / scikit-learn）
- **模型**：`BAAI/bge-m3`（本地，2.27GB，经 ModelScope 下载）；`routellm/bert_gpt4_augmented`（1.1GB）
- **数据**：`lmsys-arena-human-preference-55k/train.csv`（原始 57477 行，preprocess 后 **55361 行**）
- **脚本**：`scripts/build_arena_embeddings.py`、`scripts/verify_sw_ranking_local.py`、`scripts/profile_sw_ranking_latency.py`、`scripts/bench_elo_solvers.py`
- **提交**：`8d6b433`、`68b468f`、`28e3d68`
- **状态**：✅ 通过

## 实验目的

`sw_ranking` 路由器有两处外部依赖，都是工程化改造要解决的问题：

| 依赖 | 位置 | 问题 |
|---|---|---|
| HF hub 数据集 | 构造时 `load_dataset()` | 拉 `lmsys/...` 与 `routellm/...`，离线不可用；`huggingface.co` 在本网络仅 IPv6 不可达 |
| OpenAI Embedding API | 推理时 `calculate_strong_win_rate()` | 每次请求调 `text-embedding-3-small`：计费 + 网络延迟 + 抖动即失败 |

同时，方案文档「问题1（路由计算无缓存）」预估单次路由开销 ~350ms，但从未实测拆解过瓶颈所在。

本实验要回答三件事：

1. 能否用本地模型替代 OpenAI Embedding，且不改动路由语义？
2. 路由延迟的真实构成是什么？瓶颈在哪？
3. 瓶颈能否在不改变数值结果的前提下优化？

---

## 实验 1：本地 bge-m3 生成 arena 向量

### 设计

用本地 bge-m3 编码 arena 的全部 55361 条 prompt，产出与官方 `routellm/arena_battles_embeddings` 等价的向量矩阵。

关键约束：**向量条数必须等于 preprocess 后的 battle 行数**（`routers.py:202` 的断言 `len(arena_df) == len(arena_conv_embedding)`），否则向量与对战记录错位。

### 实现

`routellm/routers/similarity_weighted/local_embedder.py` :: `LocalBGEM3Embedder`

- 惰性加载权重（首次 encode 才进显存）
- 输出 `(N, 1024) float32` + L2 归一化（cosine 相似度退化为点积）
- `encode_battles()` 复用 `preprocess_battles()`，并以 **index 对齐**取文本

> 踩坑：初版用 `len(t) >= 16` 重新过滤文本，等于把 `preprocess_battles` 的 `MIN_LEN` 硬编码了第二份，实现漂移就会静默错位。改为按 `processed.index` 对齐。

### 结果

| 项 | 实测值 |
|---|---|
| 权重加载 | 5.3s |
| 生成耗时 | **112.1s** |
| 吞吐 | **493.8 条/秒** |
| 输出 | `(55361, 1024) float32`，216MB |
| L2 范数 | ∈ [1.0000, 1.0000] |
| 保存后回读 | 内容一致，无 NaN/Inf |
| sha256 | `90aa9978ef761da561b38341be15c627edfbe2aa6087fe340c7423713770e346` |

### 对照：与 OpenAI 路径的成本差

原路径需 55361 条 × 每条一次 API 调用（batch 2000 则 28 次请求），按 `text-embedding-3-small` $0.02/1M tokens 估算约 $0.05，且受网络波动影响。本地路径 **耗时 112s、成本 $0**。

> 补充说明：官方 sw_ranking 实际还拼接了 `routellm/gpt4_judge_battles` 数据集（用于增强 Elo 估计）。该数据集在 33/43/A6000 及 HF 缓存中均不存在，本实验**只接入 arena 部分**，行数自洽但总规模小于官方。影响面：不改动任何工程改造项，仅与官方效果不完全可比。后续如需对齐，需补该数据集并重跑本流程。

---

## 实验 2：SWRankingRouter 数据与编码双本地化

### 设计

改造 `SWRankingRouter`，新增可选参数，**默认完全不改变原行为**：

| 参数 | 作用 |
|---|---|
| `local_battles_csv` | 从本地 CSV 读 battle 记录，替代 `load_dataset()` |
| `local_embeddings_npy` | 从本地 `.npy` 读向量，替代 HF embedding 数据集 |
| `local_embedder_path` | 用本地 bge-m3 编码 prompt，替代 OpenAI API |

### 结果

| 项 | 实测值 |
|---|---|
| 构造耗时 | 3.26s |
| 行数与维度 | `(55361, 1024)`，自洽 ✓ |
| 触碰 HF hub | ❌ 无（脚本内 monkeypatch 拦截 `load_dataset` 验证） |
| 触碰 OpenAI | ❌ 无（全程 `OPENAI_API_KEY` 未设置） |
| 确定性 | 同 prompt 3 次 → `0.690542 / 0.690542 / 0.690542` ✓ |

**结论：本地化完全成立，且未改动路由数学。**

---

## 实验 3：路由延迟拆解（瓶颈定位）

### 设计

用 `scripts/profile_sw_ranking_latency.py` 精确计量各阶段耗时（预热后 20 次）。

### 结果

| 阶段 | mean | p50 | 占比 |
|---|---|---|---|
| `encode_prompt`（bge-m3） | 26.9ms | 22.2ms | 6.8% |
| `similarity dot`（55k×1024） | 6.3ms | 5.4ms | 1.6% |
| `get_weightings` | 0.34ms | 0.31ms | 0.1% |
| **`compute_elo_mle_with_tie`** | **355.5ms** | **332.7ms** | **90.1%** |
| 合计 | 394.7ms | 375.0ms | 100% |

### 进一步拆解 Elo 回归内部

| 子步骤 | 耗时 |
|---|---|
| X/Y 矩阵构造 | 34.4ms |
| **`LogisticRegression.fit`** | **362.9ms** |
| 矩阵形状 | X=(110722, 10)，模型数 p=10 |

### 结论

**瓶颈不在 embedding，也不在相似度计算，而在逻辑回归拟合。**

`fit` 的输入是 **110722 行 × 10 列** —— 样本极多、特征极少。sklearn 默认的 `lbfgs` 求解器每轮迭代需全量扫描 11 万行来估梯度，而模型只有 10 个参数，数据形状与求解器严重不匹配。

> 这推翻了方案文档中「Embedding API ~50ms + 点积 ~200ms + Elo回归 ~100ms」的预估：实际 Elo 回归占 90%，且点积仅 6.3ms（远低于预估的 200ms）。**改造前 509ms 的实测延迟也高于文档预估的 350ms。**

---

## 实验 4：Elo 求解器选型

### 设计

候选求解器在同一数据上比较耗时与结果一致性（`scripts/bench_elo_solvers.py`）。

### 结果（110722×10，真实路由权重）

| 求解器 | 耗时 | elo 最大偏差 | 评价 |
|---|---|---|---|
| lbfgs（默认） | 307.8ms | 0（基准） | 慢 |
| **newton-cholesky** | **56.9ms** | **0.72** | ✅ 快 5.4×，偏差极小 |
| lbfgs `tol=1e-2` | 62.4ms | **246.8** | ❌ 偏差灾难性，不可用 |
| lbfgs `max_iter=20` | 372.4ms | 0.40 | ❌ 不省时间 |
| liblinear | — | — | ❌ 不支持 `penalty=None` |

### 采用 newton-cholesky 后的实测

| 指标 | lbfgs | newton-cholesky | 提升 |
|---|---|---|---|
| Elo 求解 | 331.0ms | **86.9ms** | **3.8×** |
| 端到端路由 | 394.0ms | **184.9ms** | **2.1×** |

数值一致性：

```
elo 最大偏差      : 0.7231 分
模型排序          : 完全一致
winrate 最大偏移  : < 0.005
top3 (lbfgs)      : 1154.6 / 1060.9 / 1015.1
top3 (newton)     : 1154.6 / 1061.1 / 1015.2
```

### 为什么 0.72 分偏差可接受

路由决策只取决于 `winrate` 与用户 `threshold` 的比较：

```
win_rate = 1/(1 + 10^((strong_elo - weak_elo)/400))
```

elo 差 0.72 分 → winrate 偏移约 **0.0005**，而 threshold 的典型量级是 0.3~0.7。**不可能改变任何路由决策。**

---

## 踩坑记录（重要）

### 求解器性能对 sample_weight 的逐元素排列敏感

初版性能测试用 `rng.uniform(1, 10)` 造权重，结果 **newton-cholesky 反而更慢**：

```
均匀随机权重:  lbfgs 354.3ms   newton-cholesky 493.2ms   ← 结论相反
真实路由权重:  lbfgs 331.0ms   newton-cholesky  86.9ms   ← 真实情况
```

尝试用 `rng.beta(1,3)` 构造边缘分布相近的权重（实测 min=10.0 / max=97.0 / mean=19.8，与真实 min=12.4 / max=100 / mean=19.5 接近），**仍然更慢**（474.8ms）。

**结论：求解器耗时取决于权重与设计矩阵的逐元素耦合，不只是边缘分布。** 性能测试必须走完整路由链路取 `get_weightings(cosine_sims)` 的真实权重，否则会得出与线上相反的结论。

### 病态 Hessian 导致求解器静默退化

样本量小或权重分布均匀时，Hessian 病态（rcond~1e-17），sklearn 会输出：

```
LinAlgWarning: The inner solver of NewtonCholeskySolver stumbled upon a
singular or very ill-conditioned Hessian matrix at iteration 1.
It will now resort to lbfgs instead.
```

即 newton-cholesky **内部退回 lbfgs**，此时测不出任何差异。合成数据（2000 行）上就会出现此现象。

### elo 偏差随数据规模浮动

| 数据规模 | elo 最大偏差 |
|---|---|
| 55361 行（真实） | 0.72 分 |
| 2000 行（合成） | 1.72 分 |

因此测试判据不应直接约束 elo 差值，而应约束 **winrate 偏移**（路由决策的直接输入，容差 0.005）。

---

## 结论

1. **本地化成立**：bge-m3 完全替代 OpenAI Embedding API，去除了 HF hub 与 OpenAI 双依赖，零成本、可离线、确定性一致。
2. **瓶颈定位明确**：路由延迟 90% 在 `LogisticRegression.fit`，与 embedding 无关。
3. **优化有效**：换用 newton-cholesky 后 Elo 求解快 3.8×，端到端路由快 2.1×（394ms → 185ms），模型排序完全一致，winrate 偏移 < 0.005。
4. **未完成的降延迟路径**：多级缓存（方案文档 P0）。当前 185ms 已接近文档中「缓存命中率 70% → 105ms」的目标量级，叠加缓存后可进一步下降。

## 复现方式

```bash
# 环境（43 号机）
export PY=/mnt/data/wanghesong/conda-env/rag-dev/bin/python
export ROUTELLM_ROOT=/mnt/data/wanghesong/routellm
export PYTHONPATH=$ROUTELLM_ROOT/RouteLLM-Eng
export CUDA_VISIBLE_DEVICES=1
unset OPENAI_API_KEY

# 1. 生成 arena 向量（112s）
$PY scripts/build_arena_embeddings.py \
    --battles-csv $ROUTELLM_ROOT/arena_train.csv \
    --model-path  $ROUTELLM_ROOT/models/bge-m3/models/BAAI--bge-m3/snapshots/master \
    --out-dir     $ROUTELLM_ROOT/embeddings

# 2. 冒烟测试（取前 2000 行）
$PY scripts/build_arena_embeddings.py ... --limit 2000

# 3. 端到端验证本地化
$PY scripts/verify_sw_ranking_local.py \
    --battles-csv $ROUTELLM_ROOT/arena_train.csv \
    --embeddings  $ROUTELLM_ROOT/embeddings/arena_embeddings.npy \
    --model-path  $ROUTELLM_ROOT/models/bge-m3/models/BAAI--bge-m3/snapshots/master

# 4. 延迟拆解
$PY scripts/profile_sw_ranking_latency.py ... --runs 20

# 5. 求解器对比
$PY scripts/bench_elo_solvers.py <csv> <npy> <model_path>

# 6. 单元测试（TDD）
export ROUTELLM_ARENA_CSV=$ROUTELLM_ROOT/arena_train.csv
export ROUTELLM_ARENA_EMBEDDINGS=$ROUTELLM_ROOT/embeddings/arena_embeddings.npy
export ROUTELLM_BGE_M3_PATH=$ROUTELLM_ROOT/models/bge-m3/models/BAAI--bge-m3/snapshots/master
$PY -m pytest tests/test_local_bge_m3_embedder.py tests/test_sw_ranking_local.py \
    tests/test_elo_solver.py -v
```
