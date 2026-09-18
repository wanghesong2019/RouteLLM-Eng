# 🚀 RouteLLM-Eng

> 基于 LMSYS [RouteLLM](https://github.com/lmsys/routellm) 的生产级 LLM 路由网关

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-315%20passed-brightgreen.svg)](#测试)
[![Docker](https://img.shields.io/badge/docker-675MB-blue.svg)](#快速开始)

**把 LMSYS RouteLLM 从学术原型改造为生产级路由网关。** 在保持路由有效性的前提下，补齐企业级的缓存、熔断、全链路可观测性与异步高并发架构。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| ⚡ **性能** | 定位并重构 `LogisticRegression.fit` 瓶颈（改 `newton-cholesky`），路由延迟 394ms → 185ms。win-rate 多级缓存命中将延迟降低 4 个数量级。 |
| 🛡️ **容错** | 三态熔断器 + 指数退避重试 + 四级降级链（强 → 弱 → 缓存 → 503）。网关不会级联崩溃。 |
| 📊 **可观测性** | 非侵入式 FastAPI 中间件采集请求级指标 → SQLite → 内置 ECharts 面板。指标重启不丢失。 |
| 🔄 **异步** | Router 基类全面异步化 + `httpx.AsyncClient` 连接池复用。慢请求不再阻塞事件循环。 |
| ⚙️ **热更新** | 强弱模型 `base_url`、`api_key`、模型名支持运行时更新，无需重启。不可变配置对象原子替换保证并发安全。 |
| 🔒 **安全** | `/v1/models` 端点；API key 中间件（Bearer + HMAC 防时序侧信道）；运维端点严格白名单。 |

## 📐 架构

```
              ┌────────────────────────────────┐
              │  Client (OpenAI-compatible)     │
              │  base_url + api_key             │
              └──────────┬─────────────────────┘
                         │ OpenAI API + Bearer key
                         ▼
┌──────────────────────────────────────────────────────────┐
│                RouteLLM Gateway (:6060)                  │
│                                                          │
│  ApiKey Middleware → Metrics Middleware → FastAPI Server │
│  /v1/chat/completions  /metrics  /dashboard             │
│                                                          │
│  ConfigStore (hot reload) ←→ Controller (lock-free read) │
│  SQLite (metrics persistence)                            │
│  MultiTierCache (L1 LRU → L2/L3 injectable)              │
│  Routers: remote_bert (host:6070) / random              │
└──────┬───────────────────────────┬──────────────────────┘
       │                            │ read-only mount
       ▼                            ▼
┌──────────────┐          ┌──────────────────────────┐
│ Strong/Weak  │          │ Dashboard (:8092)        │
│ LLM (OpenAI  │          │ HTML + ECharts           │
│ compatible)  │          │ + Config editor tab      │
└──────────────┘          └──────────────────────────┘
```

## 快速开始

### Docker（推荐）

```bash
cp .env.example .env       # 填入强弱模型与凭据
docker compose up -d       # 网关 :6060 + 面板 :8092
```

网关暴露 OpenAI 兼容 API —— 任何兼容客户端均可接入：

```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer $ROUTE...KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"..."}]}'
```

`model` 字段是**路由规格**（`router-<name>-<threshold>`），不是下游模型名。

### 从源码运行

```bash
pip install -e ".[serve,eval]"
python -m routellm.openai_server --routers random  # random 不需要 GPU
```

> **无缝替换**：网关保持 OpenAI 兼容协议与上游接口不变，从 OpenAI 或原生 RouteLLM 切换过来只需改 `base_url`，零代码改动。

## 为什么选择 RouteLLM-Eng？

上游 RouteLLM 是学术原型——保证算法正确性，但不考虑生产环境关切：

| 问题 | 上游 | RouteLLM-Eng |
|---------|----------|-------------|
| 无缓存 | 每次请求重算 win-rate（~350ms） | 多级缓存，命中 0.02ms（4 个数量级） |
| 无可观测性 | `logging.info` + 内存 dict | FastAPI 中间件 → SQLite → ECharts 面板 |
| 无容错 | 裸调 `litellm.completion()`，无重试/超时 | 熔断器 + 重试 + 四级降级链 |
| 同步阻塞 | FastAPI 异步但路由器同步 | 全面异步：`httpx.AsyncClient` + `asyncio.to_thread` |
| 无部署 | 无 Dockerfile/CI | Docker（675MB，不含 torch）+ compose 双容器 |
| 配置固化 | 改模型需重启 | 运行时热更新，配置原子替换 |
| 默认即坏 | 默认模型 provider 已被 LiteLLM 移除 | 全部配置走环境变量，fail-fast 校验 |

## 评测

复现论文指标（APGR / CPT 框架，RouteLLM, ICLR 2025）：

| 数据集 | APGR | 95% CI | CPT(50%) |
|---------|------|--------|----------|
| MMLU | 0.5328 | [0.5157, 0.5496] | 44.30% |
| GSM8K | 0.5294 | [0.4975, 0.5646] | 45.34% |

随机基线 APGR ≈ 0.5 —— **APGR > 0.5 才算路由有效**。

> **反直觉发现**：APGR 不能用来选阈值。PGR 随走强比例单调递增，最大化 APGR 的答案永远是「全走强」。应改用 **CPT**（达到目标 PGR 所需的最小走强比例）——先定质量目标，再反解成本。

## 文档

| 路径 | 内容 |
|------|---------|
| `docs/CHANGELOG.md` | 工程日志：每一步、实测数据、踩坑记录 |
| `docs/decisions/` | 架构决策记录（ADR） |
| `scripts/README.md` | 脚本索引 |
| `services/README.md` | 推理服务部署 |

## 开源卫生

内置守卫脚本检查凭据、内网 IP 与部署产物：

```bash
python scripts/check_open_source_hygiene.py        # 扫描工作区
python scripts/check_open_source_hygiene.py --all  # 同时扫描 git 历史
```

## 测试

```bash
pytest tests/ -q
# 315 passed, 17 skipped
```

## 🗺️ 路线图

- [ ] 多路由器策略动态切换（BERT、Embedding 等）
- [ ] Prometheus / Grafana 标准指标导出
- [ ] 流式响应路由优化
- [ ] MT-Bench 评测
- [ ] 面板双语化（i18n）

## 🤝 贡献指南

欢迎贡献！提交 PR 前请：

1. 运行 `python scripts/check_open_source_hygiene.py` —— 确保无敏感信息
2. 确保 `pytest tests/ -q` 通过
3. 参考 `docs/decisions/` 下的 ADR 了解架构背景

## 许可证

MIT License（继承自上游），见 `LICENSE`。

## 引用

```bibtex
@inproceedings{ong2025routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M Waleed and Stoica, Ion},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
