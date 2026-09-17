# scripts/

一次性验证脚本与运维辅助脚本。**不属于 `routellm` 包，不参与打包。**

约定：

- 脚本顶部注明用途、用法、以及实测结果
- 结果与结论记录在 `docs/experiments/`，脚本本身只负责可复现
- 命名：`verify_*.py` 用于验证类，其他按用途命名

## 清单

| 脚本 | 用途 |
|---|---|
| `verify_bert_basic.py` | BERT 路由模型加载、基础 win rate 计算、确定性、延迟 |
| `verify_bert_discrimination.py` | 在 GSM8K / MMLU 真实数据上验证 win rate 区分度 |
| `build_arena_embeddings.py` | 用本地 bge-m3 生成 arena 55361 条 prompt 向量（替代 OpenAI Embedding API） |
| `verify_sw_ranking_local.py` | 验证 SWRankingRouter 本地化改造：本地数据+bge-m3 端到端、无 HF/OpenAI 依赖、确定性、延迟 |
| `profile_sw_ranking_latency.py` | 拆解路由延迟构成，定位瓶颈阶段（编码/点积/Elo 回归） |
| `bench_elo_solvers.py` | 对比 elo 回归各求解器的耗时与结果一致性 |
| `diagnose_sw_ranking_distribution.py` | 诊断 sw_ranking 的 win_rate 分布并标定 threshold（对比 bert） |
| `trace_sw_ranking_pipeline.py` | 逐环节追踪 sw_ranking 推理链路的中间量，定位区分度丢失环节 |
| `probe_elo_weight_sensitivity.py` | 探测 Elo 回归对样本权重的敏感度（尺度不变性 / top-K / SCALE） |
| `probe_weight_mapping.py` | 对比不同权重映射下的区分度，寻找候选修复方案 |

## 用法

```bash
# 基础验证
python scripts/verify_bert_basic.py --model <ckpt_path>

# 区分度验证 — GSM8K（同质数据）
python scripts/verify_bert_discrimination.py --model <ckpt> \
    --csv routellm/evals/gsm8k/gsm8k_responses.csv

# 区分度验证 — MMLU（跨学科，推荐）
python scripts/verify_bert_discrimination.py --model <ckpt> \
    --csv-dir routellm/evals/mmlu/responses

# arena 向量生成（本地 bge-m3）
python scripts/build_arena_embeddings.py \
    --battles-csv <arena_train.csv> \
    --model-path  <bge-m3 本地权重目录> \
    --out-dir     <输出目录>

# 冒烟测试（取前 N 行快速验证链路）
python scripts/build_arena_embeddings.py ... --limit 2000
```

依赖：`transformers`、`torch`、`numpy`（GPU 可选，CPU 也能跑但慢）。
`build_arena_embeddings.py` 额外需要 `sentence-transformers`、`pandas`。
