# 实验记录：Docker 化部署与全链路验证

- **日期**：2026-09-16
- **机器**：43 号机（4×RTX 4090）—— 部署环境；33 号机 —— 镜像构建
- **镜像**：`routellm-eng:dev` 675MB（33 构建 → docker save → 传输 → 43 load）
- **代码**：commit `8a70c88`
- **状态**：✅ 通过

## 实验目的

验证 Docker 化后的完整部署形态：
容器（RouteLLM 网关）→ host 推理服务 → 路由决策 → 下游 LLM。

## 部署架构

```
43 号机（115.233.223.42）
├── host
│   └── 推理服务 :6070  (services/inference_server.py, xlm-roberta, RTX 4090)
└── docker
    └── routellm-deploy :6060
        ├── --add-host=host.docker.internal:host-gateway
        └── ROUTELLM_INFERENCE_URL=http://host.docker.internal:6070
```

**为何都部署在 43**：43 防火墙仅开放 SSH 端口，33 无法直连 43:6070。
（详见 docs/experiments/2026-09-16-remote-router-e2e.md）

## 镜像构建（33 号机）

```
镜像大小: 675MB
构建耗时: 约 8 分钟
```

**关键**：容器内**不含 torch / transformers / datasets**（验证通过）。
依赖惰性导入使镜像从 3GB+ 降到 675MB。

验证：
```bash
$ docker run --rm routellm-eng:dev python -c "
for m in ('torch','transformers','datasets'):
    try: __import__(m); print(m, '存在')
    except ImportError: print(m, '不存在')
"
torch: 不存在
transformers: 不存在
datasets: 不存在
```

## 部署流程

```bash
# 1. 33 上构建并导出
docker build -t routellm-eng:dev .
docker save routellm-eng:dev -o routellm-eng-dev.tar   # 669MB

# 2. 传输（75 秒）
scp routellm-eng-dev.tar 43:/mnt/data/wanghesong/routellm/

# 3. 43 上加载（14 秒）
docker load -i /mnt/data/wanghesong/routellm/routellm-eng-dev.tar

# 4. 启动（key 经 env-file 传递，不落磁盘不留进程列表）
docker run -d --name routellm-deploy \
  --add-host=host.docker.internal:host-gateway \
  -p 127.0.0.1:6060:6060 \
  --env-file /tmp/routellm.env \
  routellm-eng:dev
```

## 验证结果

### 1. 容器与网关

| 项 | 结果 |
|---|---|
| 容器状态 | `Up (healthy)`，HEALTHCHECK 生效 |
| `GET /health` | `{"status":"online"}` |
| `GET /v1/models` | `router-remote_bert-0.5`（上游为 404，已修复） |
| 启动耗时 | 7 秒 |

### 2. 容器 → host 推理服务（关键网络验证）

```
容器内 DNS:      host.docker.internal → 172.18.0.1
容器 → :6070:    HTTP 200
返回:            arch=xlm-roberta, model_loaded=true, device=cuda
```

**结论**：容器经 `--add-host=host.docker.internal:host-gateway` 可访问 host 服务。

### 3. 全链路路由决策

下游 LLM：硅基流动（https://api.siliconflow.cn/v1）
强弱模型：`openai/deepseek-ai/DeepSeek-V4-Pro` / `openai/deepseek-ai/DeepSeek-V4-Flash`

| prompt | 路由到 |
|---|---|
| What is 1+1? | DeepSeek-V4-Flash |
| hi | DeepSeek-V4-Flash |
| Write a poem about the sea. | DeepSeek-V4-Flash |
| Prove that √2 is irrational using Galois theory... | DeepSeek-V4-Flash |
| Explain the trade-offs between consistency and availability... | DeepSeek-V4-Flash |

推理服务统计：`infer_count` 850（测试前 839，本次调用 11 次），确认请求到达 host。

**全部路由到弱模型** —— 与 MMLU 实验结论一致：BERT 路由器对单条通用 prompt
给出的 win_rate 普遍低于阈值 0.5。跨学科批量对比时区分度才明显
（见 2026-09-16-bert-router-validation.md，57 学科 corr = -0.7123）。

## 遇到的两个网络问题（43 环境特有）

### 问题 1：taisure.com 在 43 上不可达

```
43 解析 taisure.com → 115.233.223.42（43 自己的公网 IP）
43 直连 taisure:443 → 超时（本机无服务监听 443）
/etc/hosts 无该条目 → DNS 层解析结果
```

**影响**：首次链路测试失败于 `litellm.InternalServerError: Connection error`。
**处置**：下游 LLM 改用硅基流动（43 上可达，实测 401/200）。

### 问题 2：容器内 IPv6 优先导致挂起

见 Dockerfile 中 `sitecustomize.py` 强制 IPv4 的处理。

## 遗留事项

1. **taisure.com 的 DNS 劫持**：43 上该域名被解析到本机，原因未知
   （可能内网 DNS 策略）。若后续需要使用，需排查。
2. **推理服务未纳入 Docker 管理**：当前在 host 以裸进程运行
   （`python services/inference_server.py`）。后续可考虑容器化 + GPU 透传，
   或 systemd 托管。
3. **无鉴权**：网关与推理服务均未加认证，仅监听回环。生产部署需补充。

## 复现方式

```bash
# 43 上
bash restart_container.sh        # 见部署流程
python test_remote_bert_chain.py # 链路测试
```
