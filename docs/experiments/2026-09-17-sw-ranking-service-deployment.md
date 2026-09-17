# 实验记录：sw_ranking 服务化上线（host 推理服务 + 容器启用）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **环境**：`/mnt/data/wanghesong/conda-env/rag-dev`
- **脚本**：`/tmp/switch_routellm_swranking.sh`（一次性，未入仓库）
- **提交**：`bf95860`
- **状态**：⚠️ 链路打通，但发现区分度问题（见文末）

## 目的

把 sw_ranking 完整推理链路服务化为 host 侧独立服务，让容器在**保持零挂载、
675MB 轻量形态**的前提下启用 `remote_sw_ranking` 路由。

## 架构

```
┌─────────────────────────────────────────┐
│ 容器 routellm-deploy (704MB, 无 torch)   │
│   remote_bert ────────┐                 │
│   remote_sw_ranking ──┼── HTTP          │
└───────────────────────┼─────────────────┘
                        │ host.docker.internal
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐        ┌────────────────────────┐
│ host:6070        │        │ host:6071              │
│ inference_server │        │ sw_ranking_server      │
│ (BERT, 6.1ms)    │        │ bge-m3 + 55k向量 + Elo │
│                  │        │ (114ms)                │
└──────────────────┘        └────────────────────────┘
```

**为什么整条链路下沉而非只下沉 bge-m3**：sw_ranking 需要 bge-m3 权重（2.27GB）
+ arena 向量（216MB）+ battle CSV（184MB）。若后两样进容器需挂载近 400MB 且装
scikit-learn；若 bge-m3 也进容器还要装 torch —— 都会打破 ADR-001 的轻量网关设计。

## 实现要点

| 文件 | 改动 |
|---|---|
| `services/sw_ranking_server.py` | 新增。端口 6071，接口形状与 `inference_server.py` 的 `/v1/score` 一致；独立实现不 import routellm 包 |
| `routellm/routers/remote.py` | 新增 `RemoteSWRankingRouter`，**直接继承** `RemoteBERTRouter` 复用全部 HTTP/分片/错误处理 |
| `routellm/config.py` | 新增 `sw_ranking_inference_url`（bert 6070 与 sw_ranking 6071 是两个服务） |
| `routellm/openai_server.py` | `_inject_inference_url` 支持按路由分别注入 URL，未配置时回落默认 |

接口对齐带来的好处：容器侧无需新增任何调用逻辑，只换路由名。

## 部署验证

### 服务启动（host）

```
加载 bge-m3 + 55361 条 arena + Elo 分档：10.2s
device: cuda, rows: 55361, dimension: 1024
```

### 服务性能（host 直测）

| 项 | 值 |
|---|---|
| 单条（HTTP 端到端） | 113~141ms（p50 ≈ 114ms） |
| 批量 10 条 | 1021ms（平均 102ms/条） |
| 确定性 | 同 prompt 5 次全为 `0.690528` |

**数值一致性与进程内实现逐位吻合**：`/selfcheck` 返回 `0.6905275284692927`，
与 `scripts/verify_sw_ranking_local.py` 测得的 `0.6905` 一致。

### 容器切换

沿用 rename + stop 的保留式切换（deployment-archive README 第 4 条）：

| 项 | 值 |
|---|---|
| 新镜像 | `a113bf378f3c`（704MB） |
| 容器启动 | 60 秒 healthy（需加载 sw_ranking 的 55k 数据） |
| 路由 | `remote_bert,remote_sw_ranking` |
| 备份 | 容器 `routellm-deploy-prev` / `routellm-deploy-old`，镜像 `rollback-20260917` |

### 端到端验证

```
/v1/models →
  router-remote_bert-0.5
  router-remote_sw_ranking-0.5                    ✅ 两个路由均注册

remote_bert @ threshold=0.5:
  「hi」                    → Flash（弱）          ✅
  「证明√2无理数…」          → Pro（强）            ✅ 与基线一致

remote_sw_ranking @ threshold=0.5:
  「hi」                    → Pro（强）            ⚠️ 见下
```

## ⚠️ 已知问题：sw_ranking 区分度不足

### 现象

两个服务的 win_rate 对比（同一批 prompt）：

| prompt | sw_ranking (6071) | bert (6070) |
|---|---|---|
| hi | 0.6892 | 0.4007 |
| What is 1+1? | 0.6931 | **0.2970** |
| 法国首都 | 0.6905 | — |
| 量子纠缠 | 0.6916 | — |
| 证明√2无理数 | 0.6920 | 0.4533 |

- **sw_ranking**：全部挤在 **0.689~0.693**（极差 0.0039），且整体高于 0.5 阈值
- **bert**：0.297~0.453，有清晰区分度且方向正确

### 影响

sw_ranking 在 threshold=0.5 下**所有请求都路由到强模型**（0.689 > 0.5），
失去省钱意义，且与 bert 的行为不一致。

### 待查方向（未验证，按可能性排序）

1. **向量维度差异**：官方用 `text-embedding-3-small`（1536 维），本实现用
   bge-m3（1024 维）。替换 embedding 模型是否改变了相似度分布 → 待对比实验。
2. **算法固有特性**：`get_weightings(sims) = 10 * 10^(sim/max_sim)` 的分母是
   该 prompt 与 55k 中最相似那条的相似度。这种"按最大值归一化"可能使不同
   prompt 的权重分布形状趋同，从而抹平 win_rate 差异。
3. **threshold 未标定**：上游 `calibrate_threshold.py` 是干嘛用的、sw_ranking
   的合理 threshold 区间是多少 → 需查上游文档/代码。
4. **缺少 gpt4_judge_battles 数据**：官方 sw_ranking 还拼接了该数据集增强
   Elo 估计，当前只接了 arena 部分（见 2026-09-17-sw-ranking-localization.md）。

### 建议

在查清前，生产路由**应继续使用 `remote_bert`**（行为已验证与基线一致）。
`remote_sw_ranking` 已注册但不应作为默认路由。

## 当前部署状态

```
routellm-deploy        新容器 a113bf37，路由 remote_bert + remote_sw_ranking
routellm-deploy-prev   上一版（仅 remote_bert），已停止
routellm-deploy-old    更早版本，已停止
routellm-eng:rollback-20260917  镜像备份

host 服务：
  :6070  inference_server.py (BERT)          — 进程常驻
  :6071  sw_ranking_server.py (sw_ranking)   — 进程常驻
```

### 回滚方式

```bash
# 只回退容器（保留 sw_ranking 代码）
docker stop routellm-deploy && docker rename routellm-deploy routellm-deploy-sw
docker rename routellm-deploy-prev routellm-deploy && docker start routellm-deploy

# 或改用镜像备份
docker run -d --name routellm-deploy -p 127.0.0.1:6060:6060 \
    --add-host host.docker.internal:host-gateway \
    --env-file /tmp/routellm_run.env routellm-eng:rollback-20260917
```

## 遗留事项

1. **sw_ranking 区分度问题**（见上）—— 优先跟进
2. host 服务目前是 **nohup 裸进程**，未纳入 systemd/supervisor —— 机器重启后不会
   自动拉起，且容器健康检查依赖它
3. 容器 `--restart no`，未改 `unless-stopped`（compose 里的设定尚未启用）
4. 开源卫生守卫报的 33 项环境指纹（`docs/experiments/` 下的 IP/主机名）待清理
