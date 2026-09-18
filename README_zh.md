<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <h1 align="center">🚀 RouteLLM-Eng</h1>
  <p align="center">
    <strong>生产级 LLM 路由网关</strong><br>
    基于 LMSYS <a href="https://github.com/lmsys/routellm">RouteLLM</a>（ICLR 2025）
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/></a>
    <a href="#测试"><img src="https://img.shields.io/badge/tests-315%20passed-brightgreen.svg" alt="Tests"/></a>
    <a href="#快速开始"><img src="https://img.shields.io/badge/docker-675MB-blue.svg" alt="Docker"/></a>
  </p>
</p>

---

把 LMSYS RouteLLM 从学术原型改造为生产级路由网关——在保持路由有效性的前提下，补齐企业级缓存、熔断、全链路可观测性与异步高并发架构。

## ✨ 核心特性

- ⚡ **性能** — 定位并重构 `LogisticRegression.fit` 瓶颈（改 `newton-cholesky`），路由延迟 394ms → 185ms。多级缓存命中将 win-rate 查询降低 4 个数量级。
- 🛡️ **容错** — 三态熔断器 + 指数退避重试 + 四级降级链（强 → 弱 → 缓存 → 503）。网关不会级联崩溃。
- 📊 **可观测性** — 非侵入式 FastAPI 中间件 → SQLite → 内置 ECharts 面板。指标重启不丢失。
- 🔄 **异步** — Router 基类全面异步化 + `httpx.AsyncClient` 连接池复用。慢请求不再阻塞事件循环。
- ⚙️ **热更新** — 强弱模型 `base_url`、`api_key`、模型名支持运行时更新。不可变配置对象原子替换保证并发安全。
- 🔒 **安全** — API key 中间件（Bearer + HMAC 防时序侧信道）、`/v1/models` 端点、运维端点严格白名单。

## 📐 架构

<p align="center">
  <img src="assets/RouteLLM-Architectural-diagram.jpg" alt="RouteLLM-Eng 架构" width="80%"/>
</p>

## ⚡ 快速开始

### Docker（推荐）

```bash
cp .env.example .env       # 填入强弱模型与凭据
docker compose up -d       # 网关 :6060 + 面板 :8092
```

任何 OpenAI 兼容客户端均可接入：

```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer *** \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"你好！"}]}'
```

> `model` 字段是**路由规格**（`router-<name>-<threshold>`），不是下游模型名。

### 从源码运行

```bash
pip install -e ".[serve,eval]"
python -m routellm.openai_server --routers random  # random 不需要 GPU
```

> **无缝替换** — 只需改 `base_url`，零代码改动。

## 📊 评测

复现论文指标（APGR / CPT 框架，RouteLLM, ICLR 2025）：

| 数据集 | APGR | 95% CI | CPT(50%) |
|---------|------|--------|----------|
| MMLU | 0.5328 | [0.5157, 0.5496] | 44.30% |
| GSM8K | 0.5294 | [0.4975, 0.5646] | 45.34% |

随机基线 APGR ≈ 0.5 —— **APGR > 0.5 才算路由有效**。

> **反直觉发现**：APGR 不能用来选阈值（PGR 随走强比例单调递增）。应改用 **CPT** —— 先定质量目标，再反解成本。

<details>
<summary><strong>🔍 为什么选择 RouteLLM-Eng？（与上游对比）</strong></summary>

<br>

| 问题 | 上游 | RouteLLM-Eng |
|------|------|-------------|
| 无缓存 | 每次请求重算 win-rate（~350ms） | 多级缓存，命中 0.02ms |
| 无可观测性 | `logging.info` + 内存 dict | 中间件 → SQLite → ECharts |
| 无容错 | 裸调 `litellm.completion()`，无重试 | 熔断器 + 重试 + 四级降级链 |
| 同步阻塞 | FastAPI 异步但路由器同步 | 全面异步：`httpx` + `asyncio.to_thread` |
| 无部署 | 无 Dockerfile/CI | Docker（675MB）+ compose 双容器 |
| 配置固化 | 改模型需重启 | 运行时热更新，配置原子替换 |
| 默认即坏 | 默认 provider 已被 LiteLLM 移除 | 全部配置走环境变量，fail-fast |

</details>

## 📚 文档

| 路径 | 内容 |
|------|------|
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | 工程日志：每一步、实测数据、踩坑记录 |
| [`docs/decisions/`](docs/decisions/) | 架构决策记录（ADR） |
| [`scripts/README.md`](scripts/README.md) | 脚本索引 |
| [`services/README.md`](services/README.md) | 推理服务部署 |

## 🧪 测试

```bash
pytest tests/ -q
# 315 passed, 17 skipped
```

## 🔐 开源卫生

内置守卫脚本检查凭据、内网 IP 与部署产物：

```bash
python scripts/check_open_source_hygiene.py        # 扫描工作区
python scripts/check_open_source_hygiene.py --all  # 同时扫描 git 历史
```

## 🗺️ 路线图

- [ ] 多路由器策略动态切换（BERT、Embedding 等）
- [ ] Prometheus / Grafana 标准指标导出
- [ ] 流式响应路由优化
- [ ] MT-Bench 评测
- [ ] 面板双语化（i18n）

## 🤝 贡献指南

提交 PR 前请：

1. 运行 `python scripts/check_open_source_hygiene.py` —— 确保无敏感信息
2. 确保 `pytest tests/ -q` 通过
3. 参考 [`docs/decisions/`](docs/decisions/) 下的 ADR 了解架构背景

## 📄 许可证

MIT License（继承自上游）。见 [`LICENSE`](LICENSE)。

## 📎 引用

```bibtex
@inproceedings{ong2025routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M Waleed and Stoica, Ion},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
