# RouteLLM-Eng

> **上游基线**：LMSYS [RouteLLM](https://github.com/lmsys/routellm)（commit `0b64fdafe049e596a3f5657c219329f24af24198`，2024-08-11 快照）

RouteLLM-Eng 是把 LMSYS RouteLLM 从**学术原型**改造成**可部署、可观测、可容错**的生产级 LLM 路由网关的工程实现。

上游框架只保证算法正确性，不考虑生产环境：无测试、无缓存、无可观测性、无容错、无部署基建。本仓库补齐这些方面，并保持与上游接口兼容。

## 这是什么

RouteLLM 根据 query 难度，在强模型与弱模型之间动态路由，在效果与成本之间取得平衡。核心是学习一个 win-rate 预测器 `P(win_s | q)`，再通过阈值 α 决定路由到哪一侧：

```
routed_model = strong if P(win_s | q) >= α else weak
```

阈值 α 的控制语义容易记反：**α 调高 → 更难满足 → 更多走弱 → 更省成本但质量下降**。α 是成本约束的严格程度，不是质量门槛。

## 快速开始

### Docker（推荐）

```bash
cp .env.example .env       # 填入强弱模型与凭据
docker compose up -d       # 网关 :6060 + 监控面板 :8092
```

网关以 OpenAI 兼容协议暴露，任何兼容客户端可直接接入：

```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer $ROUTELLM_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"..."}]}'
```

请求体中的 `model` 字段是**路由规格**（`router-<name>-<threshold>`），不是下游模型名。

### conda / venv

```bash
pip install -e ".[serve,eval]"

# random 路由器不需要 GPU 与模型权重，可先验证链路
python -m routellm.openai_server --routers random
```

## 相对上游的改造

| 方向 | 内容 |
|------|------|
| 测试体系 | 从零建立，覆盖控制器 / 路由器 / 缓存 / 监控 / 鉴权 / 容错 / 异步化 |
| 多级缓存 | 缓存 win-rate 结果（而非仅 embedding）。定位到瓶颈是 Elo 回归（占路由延迟 90%），命中后延迟降 4 个数量级 |
| 可观测性 | FastAPI 中间件采集请求级指标 → SQLite → 自研 ECharts 面板；指标持久化，重启不丢 |
| 容错 | 三态熔断器 + 指数退避重试 + 四级降级链（强 → 弱 → 缓存 → 503） |
| 异步化 | Router 基类异步兜底 + `httpx.AsyncClient` 连接池复用，慢请求不再阻塞事件循环 |
| 部署 | Dockerfile（惰性导入，镜像不含 torch）+ compose 双容器编排 |
| 运行时配置 | 强弱模型 base_url / api_key / 模型名支持不重启热更新；不可变配置对象原子替换保证并发安全 |
| 网关能力 | 补 `/v1/models` 端点；API key 中间件（Bearer + hmac 防时序侧信道）+ 白名单运维端点 |
| 路由延迟优化 | 定位瓶颈为 `LogisticRegression.fit`，改用 `newton-cholesky` 求解器，端到端 394ms → 185ms |

## 文档

| 路径 | 内容 |
|------|------|
| `docs/CHANGELOG.md` | 改造日志：每步的实际动作、实测数据、踩坑 |
| `docs/decisions/` | 技术决策记录（ADR）：选了什么、排除了什么、为什么 |
| `scripts/README.md` | 脚本索引 |

## 上游已知缺陷（本项目修复）

| 问题 | 说明 |
|------|------|
| 默认配置已失效 | 默认弱模型的 provider 已被 LiteLLM 移除，一请求即 500 |
| 无自动化测试 | 上游两个 `test_*.py` 是 `if __name__ == "__main__"` 手工冒烟脚本，需真实 API key |
| 缺 `/v1/models` | 客户端无法预检模型 |
| `OpenAI()` 模块级实例化 | 无 key 时整个包无法 import |
| 模型名硬编码 | 写在 argparse 默认值里 |
| 能力无区分度场景 | 同质化任务（如 GSM8K 数学题）上路由收益极低，见下方评测说明 |

## 评测

复现论文指标（APGR / CPT 框架，RouteLLM, ICLR 2025）：

```bash
python scripts/eval_router_apgr.py --bert-url http://127.0.0.1:6070 ...
python scripts/calibrate_threshold.py --bert-url http://127.0.0.1:6070 ...
```

实测（MMLU 14042 题 / GSM8K 1319 题，强 GPT-4 系、弱 Mixtral 系对照数据）：

| 数据集 | APGR | 95% CI | CPT(50%) |
|--------|------|--------|----------|
| MMLU | 0.5328 | [0.5157, 0.5496] | 44.30% |
| GSM8K | 0.5294 | [0.4975, 0.5646] | 45.34% |

随机路由基线 APGR ≈ 0.5 —— **APGR > 0.5 才说明路由有效**。

**注意一个反直觉的结论**：APGR 不能用来选阈值。PGR 随走强比例单调递增（强模型整体更强），只最大化 APGR 的答案永远是「全走强」。选阈值必须用 **CPT**（达到目标 PGR 所需的最小走强比例）——先定质量目标，再反解成本。

## 开源卫生

仓库内置守卫脚本，检查凭据、内网指纹、部署留档产物：

```bash
python scripts/check_open_source_hygiene.py        # 扫工作区
python scripts/check_open_source_hygiene.py --all  # 同时扫 git 历史
```

## 测试

```bash
pytest tests/ -q
```

## 许可证

沿用上游 MIT License，见 `LICENSE`。

## 引用

```bibtex
@inproceedings{ong2025routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M Waleed and Stoica, Ion},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
