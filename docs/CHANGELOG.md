# 改造日志

> 本目录记录 RouteLLM-Eng 的改造过程、验证实验、实测数据与决策依据。
> 目的是让每一步"为什么这么做"都能被追溯 —— 包括失败的尝试和反直觉的发现。

## 目录结构

```
docs/
├── CHANGELOG.md          # 改造日志（本文件）：按时间顺序记录每一步
├── experiments/          # 验证实验：脚本 + 实测输出
│   ├── README.md         # 实验索引与结论摘要
│   └── 2026-09-16-*.md   # 单个实验记录（日期前缀）
└── decisions/            # 技术决策记录（ADR）
    └── ADR-*.md
```

## 记录原则

1. **实测优先**：写"实测得到 X"，不写"应该是 X"。没有实测支撑的结论标注为"待验证"。
2. **记录失败**：走错的弯路同样留痕（例：GSM8K 验证无区分度 → 换 MMLU 后反转），因为它们能防止后人重蹈。
3. **量化**：尽量给出数字（延迟、相关系数、样本量），而非定性描述。
4. **可复现**：实验脚本入库，附运行命令与环境。

---

## 2026-09-17

### 13. 运行时配置热更新（Phase 3.5 → Phase 3.6）

Phase 3.5 让下游 LLM 的 base_url / api_key / 模型名可运行时修改并立即生效，
无需重启网关（详见 `docs/experiments/2026-09-17-runtime-config-hot-reload.md`）。

Phase 3.6 是用户在 Dashboard 实际使用后提出的四项修正
（详见 `docs/experiments/2026-09-17-per-tier-config.md`）：

| # | 问题 | 修正 |
|---|---|---|
| 1 | 弱模型只有「模型名」，base_url / api_key 是网关级全局项 | 强弱**各自**一套 base_url / api_key；顶层保留为全局兜底（空则回落，向后兼容） |
| 2 | 模型名必须手写 `openai/` 前缀 | 配置层存**原始名**，`Controller.downstream_kwargs` 调用前统一拼前缀 |
| 3 | 「测试连接」只有一个，不知道测的是谁 | 拆成**两个按钮**（强 / 弱各一），响应回显被测模型、地址、凭据来源 |
| 4 | （用户补充）只填一个模型名时不应报错 | **单模型兜底**：未填的一档并入已填档；两侧都空才 fail-fast |

顺带修复一个既有缺陷：Dashboard 转发层把网关的**所有** `>=400` 都
转成 `HTTP 200 + proxy_error`，导致 4xx 客户端错误（如非法 tier 的 422、
鉴权失败的 401）在前端无法与"成功"区分。改为 **4xx 保留状态码透传**，
5xx / 不可达仍归一化（面板不崩的降级设计不变）。

测试：新增 43 例，全量 **238 passed, 17 skipped**。
端到端验证于 43 真实容器（含容器内直接验证兜底与前缀拼接）。

---

## 2026-09-16

### 1. 仓库拆分与基线建立

从求职仓库 `jobfinding` 中拆出本仓库，作为独立工程仓库。

- 上游基线：`lmsys/routellm @ 0b64fdafe049e596a3f5657c219329f24af24198`（2024-08-11 快照）
- 导入方式：与上游 zip 做 `diff -rq` 全量比对，确认无改动后入库
- 规模：22 个 Python 文件 / 3006 行代码 / 108 个跟踪文件 / 17MB
- 方案文档保留在 jobfinding（`projects/RouteLLM-优化改造方案.md`），本仓库承载实现

### 2. 环境与依赖验证（33 号机）

环境：Python 3.10.12 + venv，`pip install -e ".[serve]"`

实测安装版本：

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.14.0 | 核心依赖（非 optional） |
| transformers | 5.17.0 | 上游代码为 2024-08，跨 3 个大版本 |
| datasets | 5.0.1 | `load_dataset` 行为有变动风险 |
| litellm | 1.101.0 | |
| numpy | 1.26.4 | 符合 pyproject 的 `numpy<2` 约束 |

**结论**：`import` 全部通过，`transformers 5.x` 未破坏导入（详见实验 1）。

### 3. 端到端冒烟（改造前基线）

```
GET  /health               → 200 {"status":"online"}     ✓
GET  /v1/models            → 404                          ✗ 端点缺失
POST /v1/chat/completions  → 500                          ✗ 见下
```

**500 根因**：上游默认弱模型 `anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1` 已失效 —— LiteLLM 移除了 `anyscale` provider。故障链：默认模型名失效 → `get_llm_provider` 抛 `BadRequestError` → `controller.py:153` 裸调用无 try/except → 冒泡为 500。

换成可用 provider 后：

```
POST /v1/chat/completions  → 200 SUCCESS
  routed model : qwen3.5-flash  (random 路由器, threshold 0.5 → 弱模型)
  response     : "ROUTELLM OK"
  usage        : 313 tokens
```

### 4. 测试基建现状

上游**无可运行的自动化测试**：

```bash
$ pytest routellm/tests/ -v
collected 0 items
```

两个 `test_*.py` 的全部逻辑在 `if __name__ == "__main__":` 块内，是手工冒烟脚本，且需真实 API key。`pyproject.toml` 未声明测试依赖。

→ 改造的测试体系需从零建立，已作为 P0 前置项记入方案文档。

### 5. 推理模型下沉至 host（架构决策）

**背景**：上游 `BERTRouter` / `CausalLLMRouter` 在进程内用 transformers 加载模型。若全部塞入容器，镜像会包含 CUDA torch + 模型权重。

**决策**：模型推理下沉到 host（43 号机）独立 FastAPI 服务，RouteLLM 容器通过 HTTP 调用。推理服务独立实现，不 import `routellm` 包，保持解耦。

理由与排除的备选方案见 `decisions/ADR-001-inference-service-on-host.md`。

### 6. BERT 路由模型验证（43 号机）

模型：`routellm/bert_gpt4_augmented`（1.1GB，实际架构为 **xlm-roberta**，非 BERT）

| 项 | 实测值 |
|---|---|
| 加载耗时 | 1.2s |
| 单条推理延迟 | 6.1ms（平均，RTX 4090，预热后） |
| 确定性 | 同输入两次结果完全一致 ✓ |
| 标签数 | 3 |

**区分度验证**（详见实验 2、3）：

- GSM8K（1319 题，同质数学题）→ corr ≈ 0，**无区分度**
- MMLU（57 学科）→ corr(weak_acc, win_rate) = **-0.7123**，**区分度强**

→ 结论：模型加载正确，路由器有效性依赖"query 文本能反映难度"这一前提。同质化数据集不适用。**MMLU 应作为主要验证集，GSM8K 不适用。**

### 7. 发现的问题清单（已同步至方案文档）

| # | 问题 | 位置 | 严重度 |
|---|---|---|---|
| 1 | 上游默认模型失效，开箱即 500 | `openai_server.py` argparse 默认值 | 高 |
| 2 | `OpenAI()` 模块级实例化，无 key 时整个包无法 import | `similarity_weighted/utils.py:11` | 高 |
| 3 | `/v1/models` 端点缺失（404） | `openai_server.py` | 中 |
| 4 | 无任何自动化测试 | `routellm/tests/` | 高 |
| 5 | Pydantic V1 风格 `@validator`（6 处），V3 将移除 | `causal_llm/prompt_format.py` | 低 |
| 6 | 下游模型名需 provider 前缀，否则 litellm 报错 | 配置层 | 中 |

### 8. 推理服务实现与验证

`services/inference_server.py`（独立实现，不 import `routellm` 包）

四个端点：`POST /v1/score`（批量评分，主接口）、`GET /health`、`GET /v1/models`、`GET /selfcheck`

**实测（43 号机，RTX 4090）**：

| 指标 | 结果 |
|---|---|
| 与上游一致性 | `/selfcheck` → `matches: true`（浮点误差 < 1e-9） |
| 单条延迟（含 HTTP） | p50 10.7ms / p95 16.0ms |
| HTTP 往返开销 | ≈ 4.6ms（纯推理 6.1ms） |
| Batch 加速 | batch=128 → **68.78x**（10.18ms → 0.15ms/item） |
| MMLU 14000 题推算 | 1.3s（batch=500 × 28 批）；单条请求需 142s |
| 并发吞吐 | 368.9 items/s（10 并发 × 4 prompt） |
| 错误处理 | 空列表 / 缺字段 → HTTP 422（Pydantic 校验） |

模型支持：`bert` 已实现验证；`causal_llm` 接口占位（权重 17GB，待接入）。

详细记录见 `docs/experiments/2026-09-16-inference-api-validation.md`。

### 9. 链路打通：RemoteBERTRouter

新增 `routellm/routers/remote.py`，实现上游 `Router` 抽象接口，通过 HTTP 调用组件 8 的推理服务。
注册为 `remote_bert`（延迟导入避免循环依赖）。

**TDD**：先写 19 个测试确认 RED（`ModuleNotFoundError`），实现后全绿。

**过程中确认的两件事**：

1. `Controller.route()` **不记录** `model_counts`，只有 completion 路径记录（上游行为）
2. **上游依赖缺陷**：`batch_calculate_win_rate` 对 `NO_PARALLEL = False` 的路由器
   会调 `prompts.parallel_apply()`，但 `pandarallel` 仅在 `eval` extra 中，非核心依赖。
   上游 `random` 路由器走评测路径必 `AttributeError`。
   → `RemoteBERTRouter` 设 `NO_PARALLEL = True`（HTTP 场景下进程池无收益）。

**端到端验证（33 → SSH 隧道 → 43:6070）**：

| 项 | 结果 |
|---|---|
| 跨机器 win_rate 一致性 | 3 例逐位一致（0.2970 / 0.4007 / 0.1993） |
| 路由决策 | Controller + `remote_bert` 正常决策 |
| 批量（经隧道） | 100 条 114.2ms（1.14ms/条） |
| 错误处理 | 服务不可达 → `RemoteInferenceError` ✓ |

**重要发现 — 影响 Docker 化部署位置**：

43 号机防火墙**仅开放 SSH 端口**（20007/20031），6070/8090 从外部不可达。

```
33 → 43 探测:  22 filtered | 20007 OPEN | 20031 OPEN | 6070 filtered | 8090 filtered
```

"容器在 33、推理在 43"的方案**不可行**（网络隔离）。→ **Docker 化时两者都部署在 43**：
推理服务在 host，RouteLLM 容器经 `host.docker.internal:6070` 访问。

详细记录见 `docs/experiments/2026-09-16-remote-router-e2e.md`。

### 10. 测试体系（从零建立）

```
tests/
├── __init__.py
├── test_remote_bert_router.py    12 个：契约 / 行为 / 错误处理
└── test_router_registration.py    7 个：注册 / Controller 集成
```

运行：`pytest tests/ -v` → **19 passed**

这是本仓库第一个可运行的自动化测试体系（上游 `routellm/tests/` 是手工脚本，收集 0 items）。

### 11. Docker 化部署（问题5 改造）

**新增容器化文件**：`Dockerfile`、`docker-compose.yml`、`requirements-gateway.txt`、`.dockerignore`、`.env.example`

**镜像**：675MB（33 号机构建 → `docker save` → 传输 → 43 `docker load`）

关键：容器内**不含 torch/transformers/datasets**（惰性导入生效），镜像从 3GB+ 降到 675MB，构建从近 1 小时降到约 8 分钟。

**配置外置化**（新增 `routellm/config.py`）：
- 全部配置来自环境变量，不再硬编码默认模型名
- 启动时 `validate()`，**fail fast** —— 修复"默认模型失效导致运行时 500"
- 校验下游模型名必须带 provider 前缀

**同时修复的两个上游缺陷**：

| # | 问题 | 位置 |
|---|---|---|
| 2 | `OpenAI()` 模块级实例化，无 key 时整个包无法 import | `similarity_weighted/utils.py:11` |
| 7 | 模块级 `argparse.parse_args()`，import 时吃掉外部 argv | `openai_server.py` |

**新增 `/v1/models` 端点**（问题3，上游实测 404）

**部署验证（43 号机全链路）**：

```
网关 → 容器 → host.docker.internal:6070 推理服务 → 路由决策 → 下游 LLM
```

| 项 | 结果 |
|---|---|
| 容器状态 | `Up (healthy)`，启动 7 秒 |
| `GET /health` | `{"status":"online"}` |
| `GET /v1/models` | `router-remote_bert-0.5` |
| 容器 → host 推理服务 | HTTP 200（`host.docker.internal` → 172.18.0.1） |
| 全链路 5 个 prompt | 全部成功路由到下游并返回 |

**新发现**：43 上 `taisure.com` 被 DNS 解析到本机公网 IP（115.233.223.42），
而本机无 443 监听 → 不可达。下游 LLM 改用硅基流动（43 可达）。
详见 `docs/experiments/2026-09-16-docker-deployment-e2e.md`。

### 12. 构建过程踩坑记录（43 网络环境）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | `apt-get install` 卡 6 分钟+ | Debian 官方源不可达 | 换阿里云镜像源 → **43 秒** |
| 2 | pip 无限等待无输出 | 容器内 DNS 只返回 IPv6，本机无 IPv6 出口 | `sitecustomize.py` 强制 IPv4 |
| 3 | 无谓体积 | 误装 gcc/g++（依赖均为预编译 wheel） | 移除 |
| 4 | torch 554MB | 清华源解析到 CUDA 版 | 惰性导入 + 网关不装 torch |
| 5 | 构建上下文过大 | `.venv` 6.1GB 被打包 | `.dockerignore` 排除 |
| 6 | 43 构建极慢 | 网络差异（33 显著更快） | 改在 33 构建后传输镜像 |
