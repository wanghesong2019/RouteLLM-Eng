# 部署留档：routellm-deploy 手工容器（切 compose 前）

> 留档时间：2026-09-16 19:35
> 来源主机：43 号机（server43-X640-G40）
> 原始位置：`/tmp/routellm-deploy-archive-20260916-193554/`（43 上，重启即失）
> 留档原因：该容器是 `docker run` **手工创建**，非 compose 管理。切换到
> `docker-compose.yml` 编排前，需固化其完整启动参数，否则一旦 `docker rm`
> 则无法复原。

## 文件说明

| 文件 | 内容 |
|---|---|
| `container-summary.txt` | 人眼可读的关键参数摘要 + 等价回滚命令 |
| `env-masked.txt` | 容器环境变量。敏感值以 `len` + `sha256前12位` 打码，可校验一致性而不存明文 |
| `inspect-full.json` | `docker inspect` 完整输出（最权威的回滚依据） |
| `image-inspect.json` | 镜像 `routellm-eng:dev` 的 inspect |
| `image-summary.txt` | 镜像 ID / 大小 / 创建时间 |
| `container-logs-tail100.txt` | 容器日志末 100 行 |

**镜像 tar 未纳入本仓库**（668MB，`routellm-eng-dev.tar`），仅存于 43 的 `/tmp`
留档目录。但它是本次留档中**最关键**的产物 —— 见下文「回滚要点」。

## 关键参数（切 compose 的差异基线）

| 项 | 当前值（手工 docker run） | compose 后 |
|---|---|---|
| 镜像 | `routellm-eng:dev` (sha256:eb6f1105…ce0a8d) | 同 |
| Cmd | `python -m routellm.openai_server --host 0.0.0.0 --port 6060` | 同（由 image CMD 或 compose 指定） |
| WorkDir | `/app` | 同 |
| 端口 | `127.0.0.1:6060 -> 6060/tcp` | 同（保持仅回环） |
| 网络 | 默认 `bridge`（IP 172.18.0.2） | 独立 `routellm-net` |
| ExtraHosts | `host.docker.internal:host-gateway` | 同 |
| **RestartPolicy** | **`no`** | `unless-stopped` ← 主要改进点 |
| Privileged | false | 同 |
| 挂载 | 无 | 无 |
| 健康检查 | `curl -fsS /health`，30s/5s/3retries | 同 |
| **日志驱动** | **`json-file` 无上限** ← 长期跑会无限增长 | `json-file` 10m × 3 轮转 |
| 容器名 | `routellm-deploy` | `routellm-eng` ← 注意改名影响 |
| compose 标签 | 空（确认非 compose 管理） | 有 |

## 环境变量

```
ROUTELLM_STRONG_MODEL   openai/deepseek-ai/DeepSeek-V4-Pro
ROUTELLM_WEAK_MODEL     openai/deepseek-ai/DeepSeek-V4-Flash
ROUTELLM_API_BASE       https://api.siliconflow.cn/v1
ROUTELLM_API_KEY        len=51 sha256=13ef285d3103… （明文只在 43 运行中的容器里）
ROUTELLM_INFERENCE_URL  http://host.docker.internal:6070
ROUTELLM_ROUTERS        remote_bert
ROUTELLM_PORT           6060
```

## 回滚要点

1. **API_KEY 明文未随本仓库留档**（有意为之，且终端回显会遮蔽 key，无法读取）。
   回滚时需从**仍活着的容器**中取出：
   ```bash
   docker inspect routellm-deploy -f '{{range .Config.Env}}{{println .}}{{end}}' | grep API_KEY
   ```
   **前提是容器尚未被 `docker rm`** —— 容器删了，key 就只能去
   硅基流动控制台重新生成。

2. **镜像 tar 比 key 更要紧，但它同时含密钥痕迹**。镜像是重建不出来的（需回
   33 构建 → `docker save` → `scp` → `docker load`，且未必与原 dev 版本一致）；
   而 key 丢了可以重新申请。回滚可依赖的镜像备份位于：
   `43:/tmp/routellm-deploy-archive-20260916-193554/routellm-eng-dev.tar`
   **注意**：该 tar 为 `docker save` 全量导出，其内部镜像配置层含 79 处
   API_KEY 明文痕迹。它是二进制文件、仅存 43 本机 `/tmp`、不入 Git、不对外，
   但仍应视作含密载体 —— **勿外传、勿提交仓库、勿上传网盘/MinIO**。

3. **43 上原生的 `inspect-full.json` 已删除**。该文件由 `docker inspect` 直接
   导出，`Config.Env` 段含 API_KEY 明文（1 处），与镜像 tar 叠加构成两份
   密钥副本。本仓库保留的是**已打码版**（`<masked len=51 sha256=13ef285d3103>`），
   信息量等价（容器名 / 镜像 / Cmd / RestartPolicy / 端口 / 网络等均在），
   仅隐去明文，故删除 43 上那份无信息损失。

4. **推荐的低成本切换方式**：切 compose 时**不要 `docker rm` 旧容器**，
   而是
   ```bash
   docker rename routellm-deploy routellm-deploy-old
   docker stop routellm-deploy-old
   ```
   保留数日。这样 key 与旧容器配置都还在，回滚成本近乎为零，
   确认新版稳定后再删。

5. 等价回滚命令全文见 `container-summary.txt` 末尾（其中 `API_KEY` 需按第 1 条补齐）。
