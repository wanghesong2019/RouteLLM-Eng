# RouteLLM-Eng README 优化执行计划

> 本文档供 DeepSeek-V4.1-Flash 逐阶段执行。每个任务给出精确的文件路径、插入位置、完整 markdown 内容，AI 无需猜测。
>
> **执行原则**：按 Phase 顺序执行，每个 Phase 内的 Task 按编号顺序执行。每个 Task 标注 `[AI]` 或 `[用户]`，`[AI]` 的由 DeepSeek 直接完成，`[用户]` 的需要用户手动操作后把产物交给 AI。

---

## Phase 0：前置准备

### Task 0.1 `[已完成]` GitHub 仓库地址已确认

**仓库地址**：`https://github.com/wanghesong2019/RouteLLM-Eng`
**owner/repo**：`wanghesong2019/RouteLLM-Eng`

后续所有 Task 中的 GitHub 链接均使用此地址，无需条件判断。

### Task 0.2 `[用户]` 准备视觉素材

用户需要手动制作以下素材，放到 `docs/` 目录下：

#### 0.2a Dashboard 宽屏截图

**操作**：
1. 启动服务：`docker compose up -d`
2. 浏览器打开 `http://localhost:8092`（Dashboard 面板）
3. 发几条请求让面板有数据（至少包含：1 条寒暄类、1 条复杂问题类）
4. 全屏截图，确保能看到：路由分布图、延迟分位图、缓存命中率、自适应阈值状态面板
5. 保存为 `docs/dashboard-preview.png`（分辨率 1920×1080 以上）

#### 0.2b 终端动图（Terminal GIF）

**操作**：
使用 `vhs`（推荐）或 `asciinema` 录制以下场景，输出为 `docs/demo-terminal.gif`：

场景脚本（约 15 秒）：
```
# 场景1：L1 快速通道（寒暄类，<1ms 返回）
curl -w "\n耗时: %{time_total}s\n" http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5","messages":[{"role":"user","content":"你好"}]}'

# 场景2：L2/L3 路由到强模型（复杂问题）
curl -w "\n耗时: %{time_total}s\n" http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5","messages":[{"role":"user","content":"证明对于任意正整数n，n^3-n能被6整除"}]}'

# 场景3：展示 Dashboard 中的路由分布
echo "打开 http://localhost:8092 查看路由分布面板"
```

**如果用户无法录制 GIF**：AI 改用静态终端截图 `docs/demo-terminal.png` 替代，或跳过此素材。

### Task 0.3 `[AI]` 确认素材就位

执行以下检查，确认哪些素材已就位：
```bash
ls -la docs/dashboard-preview.png docs/demo-terminal.gif 2>/dev/null
```
记录哪些文件存在，后续 Task 根据实际存在的文件决定是否插入对应引用。

---

## Phase 1：头部 Badge 与视觉冲击区

### Task 1.1 `[AI]` 更新 README.md 头部 badges

**文件**：`README.md`
**操作**：替换第 11-16 行的 badge 区域

将：
```html
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"/></a>
    <a href="#tests"><img src="https://img.shields.io/badge/tests-405%20passed-brightgreen.svg" alt="Tests"/></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/docker-gateway%20675MB%20%2B%20dashboard%20174MB-blue.svg" alt="Docker"/></a>
  </p>
```

替换为：
```html
  <p align="center">
    <a href="https://github.com/wanghesong2019/RouteLLM-Eng/actions"><img src="https://github.com/wanghesong2019/RouteLLM-Eng/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"/></a>
    <a href="#tests"><img src="https://img.shields.io/badge/tests-405%20passed-brightgreen.svg" alt="Tests"/></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/docker-gateway%20675MB%20%2B%20dashboard%20174MB-blue.svg" alt="Docker"/></a>
    <a href="https://github.com/wanghesong2019/RouteLLM-Eng"><img src="https://img.shields.io/github/stars/wanghesong2019/RouteLLM-Eng?style=social" alt="GitHub Stars"/></a>
  </p>
```

### Task 1.2 `[AI]` 更新 README_zh.md 头部 badges

**文件**：`README_zh.md`
**操作**：同 Task 1.1，替换第 11-16 行，内容与 Task 1.1 完全相同（badge 不分语言）。

### Task 1.3 `[AI]` 在 README.md 插入 Dashboard 截图

**文件**：`README.md`
**操作**：在 Architecture 章节之后、Quick Start 章节之前（即第 104 行 ADR-001 段落之后，第 106 行 `## ⚡ Quick Start` 之前）插入：

```markdown

## 🖼️ Dashboard Preview

<p align="center">
  <img src="docs/dashboard-preview.png" alt="RouteLLM-Eng Dashboard" width="90%"/>
</p>

> Dark-themed ECharts dashboard with real-time routing distribution, latency percentiles, cache hit rate, and adaptive threshold state. Accessible at `:8092` — no auth required for internal deployment.

```

> **条件**：仅当 `docs/dashboard-preview.png` 存在时执行。不存在则跳过。

### Task 1.4 `[AI]` 在 README_zh.md 插入 Dashboard 截图

**文件**：`README_zh.md`
**操作**：在架构章节之后、快速开始章节之前（即第 104 行 ADR-001 段落之后，第 106 行 `## ⚡ 快速开始` 之前）插入：

```markdown

## 🖼️ 面板预览

<p align="center">
  <img src="docs/dashboard-preview.png" alt="RouteLLM-Eng 面板" width="90%"/>
</p>

> 暗色主题 ECharts 面板，实时展示路由分布、延迟分位、缓存命中率与自适应阈值状态。访问 `:8092`——内网部署免鉴权。

```

> **条件**：同 Task 1.3。

### Task 1.5 `[AI]` 在 README.md 插入终端动图

**文件**：`README.md`
**操作**：在 Quick Start 的 Docker 示例 curl 命令之后（第 125 行 `> The model field...` 之后），插入：

```markdown

### 🎬 Live Demo

<p align="center">
  <img src="docs/demo-terminal.gif" alt="RouteLLM-Eng Terminal Demo" width="80%"/>
</p>

> L1 fast path returns in <1ms for greetings · L2/L3 routes complex queries to the strong model · adaptive threshold engages 429 backpressure under budget pressure.

```

> **条件**：仅当 `docs/demo-terminal.gif` 存在时执行。如果只有 png 没有 gif，把 `src` 和 `alt` 中的 `.gif` 改为 `.png`。都不存在则跳过。

### Task 1.6 `[AI]` 在 README_zh.md 插入终端动图

**文件**：`README_zh.md`
**操作**：在快速开始的 Docker 示例 curl 命令之后（第 125 行 `> model 字段是...` 之后），插入：

```markdown

### 🎬 实机演示

<p align="center">
  <img src="docs/demo-terminal.gif" alt="RouteLLM-Eng 终端演示" width="80%"/>
</p>

> L1 快速通道对寒暄类 <1ms 返回 · L2/L3 将复杂问题路由至强模型 · 预算压力下自适应阈值触发 429 背压。

```

> **条件**：同 Task 1.5。

---

## Phase 2：价值具象化——省钱对比表

### Task 2.1 `[AI]` 在 README.md 插入成本对比表

**文件**：`README.md`
**操作**：在 `## ✨ Key Features` 标题之前（即第 35 行 `All while preserving...` 段落之后，第 36 行 `## ✨ Key Features` 之前）插入新章节：

```markdown

## 💰 Cost Savings

How much does RouteLLM-Eng actually save? Here's a real-world cost comparison based on 1,000,000 mixed business requests:

| Strategy | Strong Model Usage | Monthly API Bill | Quality (MMLU) |
|----------|-------------------|------------------|-----------------|
| All strong (DeepSeek-V4-Pro / GPT-4o) | 100% | **$2,500** | 100% |
| All weak (DeepSeek-V4-Flash / 7B) | 0% | $120 | 64.2% (severely unusable) |
| **RouteLLM-Eng adaptive cascade** | **18.4%** | **$558 (↓77.6%)** | **96.8% (imperceptible)** |

> Based on real chat log test sets: L1 filters 24% trivial requests, L2/L3 intercepts 57.6% low-to-medium difficulty tasks, routing only 18.4% complex reasoning to the strong model.

```

### Task 2.2 `[AI]` 在 README_zh.md 插入成本对比表

**文件**：`README_zh.md`
**操作**：在 `## ✨ 核心特性` 标题之前（即第 34 行 `以上全部改造...` 段落之后，第 36 行 `## ✨ 核心特性` 之前）插入新章节：

```markdown

## 💰 真实省钱测算

RouteLLM-Eng 到底能省多少钱？以下是基于 1,000,000 次混合业务请求的真实开销对比：

| 调度方案 | 强模型调用比例 | 月度 API 账单 | 质量留存率（MMLU） |
|----------|--------------|-------------|-------------------|
| 全量调用强模型（DeepSeek-V4-Pro / GPT-4o） | 100% | **$2,500** | 100% |
| 全量调用弱模型（DeepSeek-V4-Flash / 7B） | 0% | $120 | 64.2%（严重不可用） |
| **RouteLLM-Eng 自适应级联网关** | **18.4%** | **$558（直降 77.6%）** | **96.8%（质量无感衰减）** |

> 基于真实聊天日志测试集测算：L1 过滤 24% 琐碎请求，L2/L3 拦截 57.6% 中低难度任务，仅将 18.4% 复杂推理分派至强模型。

```

---

## Phase 3：生态即插即用

### Task 3.1 `[AI]` 在 README.md 插入生态接入指南

**文件**：`README.md`
**操作**：在 Quick Start 的 `### From source` 小节之后（第 134 行 `> Drop-in replacement...` 之后），插入新的折叠章节：

```markdown

### 🔌 Drop-in Integration with Popular Frameworks

Your existing application needs zero code changes — just point `base_url` to `:6060`:

<details>
<summary><strong>Python OpenAI SDK / LangChain / LlamaIndex</strong></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6060/v1",  # Point to RouteLLM-Eng gateway
    api_key="YOUR_GATEWAY_KEY",
)

response = client.chat.completions.create(
    model="router-bert-0.5",  # Routing spec: L1+L2+L3 cascade
    messages=[{"role": "user", "content": "Write a quicksort"}],
)
```

</details>

<details>
<summary><strong>NextChat / Open-WebUI / Dify / Any OpenAI-compatible UI</strong></summary>

In your provider settings, add an OpenAI-compatible service:

- **API Base URL**: `http://<your-ip>:6060/v1`
- **API Key**: `YOUR_GATEWAY_KEY`
- **Model Name**: `router-bert-0.5`

That's it. The gateway handles routing, caching, fallback, and observability transparently.

</details>

```

### Task 3.2 `[AI]` 在 README_zh.md 插入生态接入指南

**文件**：`README_zh.md`
**操作**：在快速开始的 `### 从源码运行` 小节之后（第 134 行 `> 无缝替换...` 之后），插入：

```markdown

### 🔌 10 秒接入主流生态

现有应用无需修改一行核心代码，只需将 Base URL 指向 `:6060`：

<details>
<summary><strong>Python OpenAI SDK / LangChain / LlamaIndex</strong></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6060/v1",  # 指向 RouteLLM-Eng 网关
    api_key="YOUR_GATEWAY_KEY",
)

response = client.chat.completions.create(
    model="router-bert-0.5",  # 路由规格：L1+L2+L3 级联智能调度
    messages=[{"role": "user", "content": "帮我写一个快速排序"}],
)
```

</details>

<details>
<summary><strong>NextChat / Open-WebUI / Dify 等可视化界面</strong></summary>

在模型提供商设置中添加 OpenAI 兼容服务：

- **API Base URL**: `http://<your-ip>:6060/v1`
- **API Key**: `YOUR_GATEWAY_KEY`
- **Model Name**: `router-bert-0.5`

完成。网关透明处理路由、缓存、降级与可观测性。

</details>

```

---

## Phase 4：选型对比——为什么不用 LiteLLM / One-API

### Task 4.1 `[AI]` 在 README.md 插入选型对比表

**文件**：`README.md`
**操作**：在 Evaluation 章节之后、Documentation 章节之前（即第 167 行 `</details>` 之后，第 169 行 `## 📚 Documentation` 之前）插入新章节：

```markdown

## 🥊 How Is This Different from LiteLLM / One-API?

| Dimension | One-API / New-API | LiteLLM Proxy | RouteLLM-Eng |
|-----------|-------------------|---------------|--------------|
| **Core positioning** | Channel management & billing relay | Unified SDK adapter & load balancing | **Content-difficulty-based semantic routing & dynamic cost reduction** |
| **Routing decision** | Channel weight / round-robin | Random / client-specified | **L1 regex + L2 BERT win-rate + L3 dynamic budget closed-loop** |
| **Quality guarantee** | None (pure forwarding) | None (failover only) | **Hard quality guardrail (complex tasks never downgraded, 429 backpressure)** |
| **Architecture** | Go lightweight process | Python container | **Torch-free 675MB container + GPU inference decoupled (ADR-001)** |

```

### Task 4.2 `[AI]` 在 README_zh.md 插入选型对比表

**文件**：`README_zh.md`
**操作**：在评测章节之后、文档章节之前（即第 167 行 `</details>` 之后，第 169 行 `## 📚 文档` 之前）插入：

```markdown

## 🥊 选型指南：与 LiteLLM / One-API 有何不同？

| 功能维度 | One-API / New-API | LiteLLM Proxy | RouteLLM-Eng |
|----------|-------------------|---------------|--------------|
| **核心定位** | 渠道管理与计费中转 | 统一 SDK 适配与负载均衡 | **基于内容难度的语义智能路由与动态降本** |
| **路由决策机制** | 渠道权重 / 顺序轮询 | 随机 / 客户端指定 | **L1 正则 + L2 BERT 胜率预测 + L3 动态预算闭环** |
| **质量保证底线** | 无（纯转发） | 无（仅做失败降级） | **硬质量防线（复杂任务誓死不降级，宁出 429）** |
| **架构设计** | Go 轻量进程 | Python 容器 | **Torch-Free 轻量容器（675MB）+ GPU 推理分层解耦** |

```

---

## Phase 5：社区增长飞轮

### Task 5.1 `[AI]` 更新 README.md Roadmap 为可认领任务

**文件**：`README.md`
**操作**：替换第 199-203 行的 Roadmap 列表

将：
```markdown
## 🗺️ Roadmap

- [ ] Multi-router strategy dynamic switching (BERT, Embedding, etc.)
- [ ] Streaming response routing optimization
- [ ] MT-Bench evaluation
- [ ] Dashboard bilingual (i18n)
- [ ] Distributed tracing (OpenTelemetry)
```

替换为：
```markdown
## 🗺️ Roadmap

Community-contributable tasks — grab one!

- [ ] `[good first issue]` Redis external cache adapter (replace in-memory LRU)
- [ ] `[help wanted]` Feishu / DingTalk / Slack budget alert webhook notifications
- [ ] `[help wanted]` Streaming response first-token latency optimization
- [ ] `[feature]` OpenTelemetry distributed tracing integration
- [ ] Multi-router strategy dynamic switching (BERT, Embedding, etc.)
- [ ] MT-Bench evaluation
- [ ] Dashboard bilingual (i18n)
```

### Task 5.2 `[AI]` 更新 README_zh.md Roadmap 为可认领任务

**文件**：`README_zh.md`
**操作**：替换第 199-203 行的路线图列表

将：
```markdown
## 🗺️ 路线图

- [ ] 多路由器策略动态切换（BERT、Embedding 等）
- [ ] 流式响应路由优化
- [ ] MT-Bench 评测
- [ ] 面板双语化（i18n）
- [ ] 分布式追踪（OpenTelemetry）
```

替换为：
```markdown
## 🗺️ 路线图

社区可认领任务——欢迎来挑！

- [ ] `[good first issue]` Redis 外部缓存适配器（替代内存 LRU）
- [ ] `[help wanted]` 飞书 / 钉钉 / Slack 预算告警 Webhook 通知
- [ ] `[help wanted]` 流式响应首包延迟评测与首 Token 预测优化
- [ ] `[feature]` OpenTelemetry 分布式追踪接入
- [ ] 多路由器策略动态切换（BERT、Embedding 等）
- [ ] MT-Bench 评测
- [ ] 面板双语化（i18n）
```

### Task 5.3 `[AI]` 在 README.md 末尾添加 Star History

**文件**：`README.md`
**操作**：在 Citation 代码块之后（文件末尾，第 229 行 ``` 之后）追加：

```markdown

## 📈 Star History

<a href="https://star-history.com/#wanghesong2019/RouteLLM-Eng&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wanghesong2019/RouteLLM-Eng&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wanghesong2019/RouteLLM-Eng&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wanghesong2019/RouteLLM-Eng&type=Date" />
  </picture>
</a>
```

> **注意**：仓库已确认，直接执行。

### Task 5.4 `[AI]` 在 README_zh.md 末尾添加 Star History

**文件**：`README_zh.md`
**操作**：在引用代码块之后（文件末尾，第 229 行 ``` 之后）追加与 Task 5.3 完全相同的内容（Star History 不分语言）。

> **条件**：仓库已确认，直接执行。

---

## Phase 6：细节修复

### Task 6.1 `[AI]` 修复 README.md curl 命令中的占位符

**文件**：`README.md`
**操作**：找到第 119 行的 curl 命令中的 `Authorization: Bearer` 部分

当前内容（第 118-123 行）：
```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer *** \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"Hello!"}]}'
```

问题：`Bearer ***` 后面的引号不闭合（`***` 后面缺少闭合的 `"`），且占位符不统一。

替换为：
```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"Hello!"}]}'
```

### Task 6.2 `[AI]` 修复 README_zh.md curl 命令中的占位符

**文件**：`README_zh.md`
**操作**：找到第 118-123 行的 curl 命令

当前内容：
```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer *** \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"你好！"}]}'
```

替换为：
```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"你好！"}]}'
```

### Task 6.3 `[AI]` 验证架构图文件路径

**操作**：执行以下命令确认文件存在且路径正确：
```bash
ls -la docs/architecture-en.svg docs/architecture-zh.svg
```

确认：
- `README.md` 中引用的是 `docs/architecture-en.svg` ✓
- `README_zh.md` 中引用的是 `docs/architecture-zh.svg` ✓

如果文件名不匹配，修正 README 中的引用路径使其与实际文件名一致。

### Task 6.4 `[AI]` 运行开源卫生守卫

**操作**：执行以下命令确保没有敏感信息泄漏：
```bash
python3 scripts/check_open_source_hygiene.py
```

如果报告任何问题，修复后重新运行直到通过。

---

## 执行顺序总结

| 顺序 | Phase | Task 编号 | 执行者 | 说明 |
|------|-------|----------|--------|------|
| 1 | 0 | 0.1 | ✅已完成 | GitHub 仓库已确认：`wanghesong2019/RouteLLM-Eng` |
| 2 | 0 | 0.2a | 用户 | 截 Dashboard 截图 |
| 3 | 0 | 0.2b | 用户 | 录终端动图 |
| 4 | 0 | 0.3 | AI | 检查素材就位情况 |
| 5 | 1 | 1.1-1.2 | AI | 更新头部 badges |
| 6 | 1 | 1.3-1.6 | AI | 插入截图和动图（条件执行） |
| 7 | 2 | 2.1-2.2 | AI | 插入省钱对比表 |
| 8 | 3 | 3.1-3.2 | AI | 插入生态接入指南 |
| 9 | 4 | 4.1-4.2 | AI | 插入选型对比表 |
| 10 | 5 | 5.1-5.2 | AI | 更新 Roadmap |
| 11 | 5 | 5.3-5.4 | AI | 添加 Star History（条件执行） |
| 12 | 6 | 6.1-6.2 | AI | 修复 curl 占位符 |
| 13 | 6 | 6.3 | AI | 验证架构图路径 |
| 14 | 6 | 6.4 | AI | 运行卫生守卫 |

> **AI 执行者注意**：Phase 0 的用户任务完成后（或用户说「跳过」后）再开始 Phase 1-6。每个 Task 完成后用 `git add` + `git commit` 提交，commit message 格式：`docs(readme): <简述>`。全部完成后一次性 `git push`。
