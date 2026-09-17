# 实验记录：多级缓存层（P0）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **脚本/模块**：`routellm/cache/`（base / keys / lru_cache / multi_tier）
- **状态**：✅ 通过

## 目标（方案文档 4.1 节，P0）

方案文档预估「缓存命中率 70% → 延迟 105ms，90% → 35ms」。
原始问题：每次请求都要重新执行 编码 + 55k 相似度 + Elo 回归。

## 相对方案文档的现实调整

方案文档写作时的瓶颈是 **OpenAI Embedding API（~50ms/次）**，故设计的
缓存对象是 **embedding 向量**。但实测（见 2026-09-17-sw-ranking-localization.md）
瓶颈已变为 **Elo 回归**（`LogisticRegression.fit` 占路由延迟 90%，355ms）。

**因此本实现缓存 `win_rate` 结果** —— 命中时连相似度计算与 Elo 回归都跳过，
这才是降延迟的关键。若只缓存 embedding，仍需跑 55k 点积 + Elo 回归（约 90% 开销）。

## 实现

### 模块结构

```
routellm/cache/
├── base.py         CacheBackend 抽象基类（async 接口）
├── keys.py         key 生成（sha256 前 16 位，区分 res/emb 前缀）
├── lru_cache.py    L1: 进程内 LRU（thread-safe，TTL，命中统计）
└── multi_tier.py   MultiTierCache：逐级查找 + 回填 + 故障降级
```

### 关键设计决策

| 决策 | 理由 |
|---|---|
| 缓存 win_rate 而非 embedding | 实测 Elo 回归占 90%，缓存结果才省掉链路 |
| key 用 hash 而非原文 | prompt 可达数 KB；避免缓存留存用户原文 |
| 分层查找 + 回填上层 | 下层命中后上层也有，下次更快 |
| 单层故障自动降级 | Redis 挂了不应导致缓存整体不可用 |
| L1 用 OrderedDict + Lock | move_to_end 实现 LRU；锁保护多线程/协程并发 |

### 接入点

| 位置 | 说明 |
|---|---|
| `SWRankingRouter.calculate_strong_win_rate` | 查缓存 → 命中返回；否则计算并回写 |
| `services/sw_ranking_server.score_batch` | 同上；支持批量（部分命中部分 miss） |

服务内的缓存为**独立实现**（`_LRUCache`）以保证服务不依赖 `routellm` 包
（可独立部署的既有约定）。key 算法一致性由测试 `test_key_algorithm_matches_routellm`
保障。

## 实测效果（43，真实数据）

### 单条请求

```
第 1 次（miss）: 606.02 ms
第 2 次（hit） :   0.04 ms
第 3 次（hit） :   0.04 ms
换 prompt（miss）: 289.71 ms
再查原 prompt（hit）: 0.03 ms
```

**miss ~290~600ms → hit 0.02~0.05ms，快约 4 个数量级。**

### 批量场景（5 重复 + 5 新）

```
elapsed: 1390.3 ms
cached 标记: [True, True, True, True, True, False, False, False, False, False]
```

重复的 5 条全部命中，新 prompt 全部 miss。缓存统计同步更新：
```
hits: 6 | misses: 6 | hit_rate: 0.5
```

### 与方案文档预估的对比

| 场景 | 文档预估 | 实测 |
|---|---|---|
| 缓存命中 | 0.01ms（L1） | **0.02~0.05ms** ✅ 吻合 |
| 未命中 | ~350ms | 290~600ms |
| 命中率 70% 时 | ~105ms | 理论：0.7×0.04 + 0.3×400 ≈ **120ms** |

**注意**：命中率取决于流量特征（重复 prompt 比例）。文档假设的 70%~90%
在真实业务下的可达性需实测，本实验只验证了机制正确性与单次命中收益。

### 可观测性

`/health` 暴露缓存统计：
```json
"cache": {"enabled": true, "hits": 6, "misses": 6, "hit_rate": 0.5,
          "size": 6, "maxsize": 4096, "backend": "lru"}
```

`/v1/score` 的每条结果带 `cached` 标记（便于排查与观测）。

## 测试

| 测试文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_cache.py` | 12 | LRU 基本/淘汰/TTL/统计、多级编排、回填、故障降级、key 生成 |
| `tests/test_router_cache.py` | 7 | 命中跳过计算、miss 后回写、位级一致、不同 prompt 不碰撞、无缓存兼容、统计暴露、故障降级 |
| `tests/test_sw_ranking_server.py` | +4 | key 一致性、服务内 LRU 行为 |

全量回归：**75 passed, 17 skipped**（本机）；43 上服务测试 **14 passed**。

关键不变量测试：
- **缓存值与直接计算值位级一致**（`test_cached_value_bitwise_identical`）——
  避免缓存引入路由决策的不确定性
- **缓存故障时路由仍正常返回**（`test_cache_failure_degrades_gracefully`）

## 未做的事

1. **L2/L3 未接线**：`MultiTierCache` 已支持多层编排并注入任意后端，
   但 Redis / SQLite 后端尚未实现（当前只用 L1）。
   接入方式：`MultiTierCache([LRUCache(...), RedisCache(...), SQLiteCache(...)])`
2. **命中率未在真实业务流量下验证**：本实验用的是合成请求序列
3. **缓存一致性**：若支撑数据（arena 向量/模型）变更，旧缓存需失效 ——
   当前依赖 TTL（24h），未做主动失效

## 复现方式

```bash
export PY=/mnt/data/wanghesong/conda-env/rag-dev/bin/python
export ROOT=/mnt/data/wanghesong/routellm

# 启动带缓存的服务（默认 cache-size=4096, ttl=24h）
cd $ROOT/RouteLLM-Eng
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 $PY services/sw_ranking_server.py \
    --model-path $ROOT/models/bge-m3/models/BAAI--bge-m3/snapshots/master \
    --battles-csv $ROOT/arena_train.csv \
    --embeddings $ROOT/embeddings/arena_embeddings.npy \
    --judge-parquet $ROOT/gpt4_judge_battles.parquet \
    --judge-embeddings $ROOT/embeddings/judge_embeddings.npy \
    --cache-size 4096 \
    --port 6071

# 验证缓存（第一次 miss，第二次 hit）
P='What is the capital of France?'
for i in 1 2; do
  curl -s -X POST localhost:6071/v1/score -H 'Content-Type: application/json' \
    -d "{\"prompts\":[\"$P\"]}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);r=d['results'][0];print(f\"{d['elapsed_ms']:.2f} ms  cached={r.get('cached')}\")"
done

# 查看缓存统计
curl -s localhost:6071/health | python3 -c "import sys,json;print(json.load(sys.stdin)['cache'])"
```
