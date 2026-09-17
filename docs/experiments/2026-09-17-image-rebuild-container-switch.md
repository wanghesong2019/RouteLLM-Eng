# 实验记录：镜像重建与容器切换（2026-09-17 代码生效验证）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **脚本**：`/tmp/switch_routellm_container.sh`（一次性，未入仓库）
- **状态**：✅ 通过，无回归

## 目的

`sw_ranking` 本地化与 Elo 求解器改造（`8d6b433` / `68b468f` / `28e3d68`）只改到了
代码仓库，43 上运行中的容器仍用旧镜像（`eb6f1105`，23 小时前构建）。

本实验验证：**用新代码重建镜像并切换容器后，服务行为与基线完全一致。**

## 切换前基线（旧镜像 eb6f1105）

| 项 | 值 |
|---|---|
| 容器 | `routellm-deploy`，镜像 `routellm-eng:dev` (eb6f1105)，healthy |
| 端口 | `127.0.0.1:6060` |
| 路由 | `remote_bert`（BERT 推理走 host 6070） |
| `/health` | `{"status":"online"}` |
| `/v1/models` | `router-remote_bert-0.5` |
| 简单问题「hi」 | 路由到弱模型 `DeepSeek-V4-Flash` |
| 复杂问题（证明 √2 无理） | 路由到强模型 `DeepSeek-V4-Pro` |
| 端到端延迟 | 1.76s ~ 16.44s（**主要是下游 LLM 生成，非路由开销**） |

### 延迟实测的重要结论

```
16.444s  3.568s  8.876s  1.762s  4.223s
```

路由开销（remote_bert 推理 6.1ms；sw_ranking 改造后 185ms）相比下游 LLM 生成
（1.7~16s）**占比不到 5%**。

→ **在单请求场景下，路由延迟优化的用户感知很弱。** 其真实价值在高并发场景：
路由是 CPU 密集的同步阻塞操作，会成为吞吐瓶颈（对应方案文档「问题4：同步阻塞」）。

## 容器切换

### 关键约束（来自 deployment-archive-2026-09-16/README.md）

1. **API_KEY 明文只存在于活着的容器里** —— 一旦 `docker rm` 就无法复原，
   需去硅基流动控制台重新申请。
2. 因此采用**rename + stop 而非 rm** 的切换方式，保留旧容器以便回滚。

### 操作步骤

```bash
# 1. 备份现有镜像（回退用）
docker tag routellm-eng:dev routellm-eng:rollback-20260917

# 2. 从活着的旧容器导出环境变量（含 API_KEY）
docker inspect routellm-deploy -f '{{range .Config.Env}}{{println .}}{{end}}' \
    > /tmp/routellm_env_backup.txt

# 3. 重建镜像
cd /mnt/data/wanghesong/routellm/RouteLLM-Eng
docker build -t routellm-eng:dev .

# 4. 切换容器（脚本：rename + stop 旧的，run 新的）
bash /tmp/switch_routellm_container.sh
```

切换脚本用 `--env-file` 传环境变量，避免 key 明文出现在命令行（`ps` 可见）
或 shell 历史中。

### 镜像变化

| 项 | 旧 | 新 |
|---|---|---|
| 镜像 ID | `eb6f1105` | `836c67d4` |
| 大小 | 675MB | **704MB**（+29MB） |
| 含 `local_battles_csv` | — | ✅ 7 处 |
| 含 `newton-cholesky` | — | ✅ 4 处 |
| 含 `local_embedder.py` | — | ✅ |
| torch | 未装 | **仍未装** ✓（保持轻量网关设计） |

镜像只增 29MB，因为 bge-m3 相关依赖（sentence-transformers/torch）**未进容器** ——
沿用了「模型推理下沉到 host」的既有架构决策（ADR-001）。

## 切换后验证（逐项对比基线）

| 验证项 | 基线 | 切换后 | 结果 |
|---|---|---|---|
| 容器状态 | healthy | healthy | ✅ |
| `/health` | `{"status":"online"}` | `{"status":"online"}` | ✅ |
| `/v1/models` | `router-remote_bert-0.5` | `router-remote_bert-0.5` | ✅ |
| 简单问题路由 | → Flash（弱） | → Flash（弱） | ✅ |
| 复杂问题路由 | → Pro（强） | → Pro（强） | ✅ |

**结论：新代码已上线，服务行为无回归。**

## 当前部署状态

```
routellm-deploy                 新容器，镜像 836c67d4（含全部改造）
routellm-deploy-old             旧容器，已停止但保留，可一键回滚
routellm-eng:rollback-20260917  旧镜像备份
```

### 回滚方式

```bash
docker stop routellm-deploy && docker rename routellm-deploy routellm-deploy-new
docker rename routellm-deploy-old routellm-deploy && docker start routellm-deploy
```

## 未完成事项

**容器当前仍只启用 `remote_bert`，sw_ranking 未启用。** 本实验只验证了「新代码
进了镜像且不破坏现有服务」，尚未让 sw_ranking 在服务上真正跑起来。

要启用 sw_ranking，需解决其运行时依赖：

| 依赖 | 体积 | 当前状态 |
|---|---|---|
| arena battle CSV | 184MB | 在 host `/mnt/data/wanghesong/routellm/` |
| arena 向量 .npy | 216MB | 同上 |
| bge-m3 权重 | 2.27GB | 同上 |

容器目前**零挂载**，看不到 host 文件。两条路径：

- **A（简单）**：挂载这三个路径进容器 + 装 `sentence-transformers` + 暴露 GPU
  → 但会打破「675MB 轻量网关」的设计，镜像体积会涨到 GB 级
- **B（贴合既有架构）**：把 bge-m3 也封装成 host 侧推理服务（对齐现有
  `services/inference_server.py`@6070 的模式），容器经 `host.docker.internal`
  调用 → 保持网关轻量，且为「问题6：provider 可插拔」铺路

**推荐 B**，理由是与 ADR-001 的架构决策一致。
