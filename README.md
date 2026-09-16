# RouteLLM-Eng

> **上游基线**：LMSYS [RouteLLM](https://github.com/lmsys/routellm)（commit `0b64fdafe049e596a3f5657c219329f24af24198`，2024-08-11 快照）
> 本仓库为工程化改造工作区，基于上游代码做生产化改造。

---

## 这个仓库是什么

RouteLLM 是 LMSYS（Chatbot Arena 团队）开源的 LLM 路由框架：根据 query 难度，在强模型（GPT-4）和弱模型（Mixtral-8x7B）之间动态路由，在效果与成本间取平衡。

上游是**学术项目**——只保证正确性，不考虑生产环境。本仓库的定位是把它改造成可部署、可观测、可容错的工程系统。

## 仓库关系

| 仓库 | 定位 | 说明 |
|------|------|------|
| `RouteLLM-Eng`（本仓库） | 工程实现 | 代码、测试、评测产物 |
| `jobfinding` | 求职材料 | 含 `projects/RouteLLM-优化改造方案.md`（改造方案文档，面试叙事） |

改造方案的**唯一权威依据**是 jobfinding 仓库中的方案文档，本仓库只承载实现。

## 开发与环境纪律

开发在三台机器上进行，职责分离：

| 机器 | 角色 | 说明 |
|------|------|------|
| GitLab（`172.17.17.50:2424`） | 远程权威 | 唯一真实的代码源 |
| **33 号机**（`icsr-ESC8000-G4`，8×RTX 3090） | 开发 + 测试闸门 | 所有代码修改在此进行，RED-GREEN 测试通过后才允许同步 |
| **43 号机**（`server43-X640-G40`，4×RTX 4090） | 执行环境（只读） | 只负责跑真实负载与长跑实验，**不允许直接改代码** |

**核心规则**：

1. 所有代码修改在 33 号机进行，遵循 RED-GREEN TDD。
2. 测试全绿后，才同步到 43 号机执行。
3. 43 号机上任何临时改动（调参、改路径、修报错）都不算数，必须搬回 33 号机走测试流程——否则两端漂移，下次同步会被覆盖。
4. 43 号机到 GitLab 网络不通（全端口 filtered），一切 pull/push 经由 33 号机中转。

## 上游原始能力

```bash
# 安装（含服务与评测依赖）
pip install -e ".[serve,eval]"

# 启动 OpenAI 兼容服务（random 路由器不需要 GPU / 模型权重）
python -m routellm.openai_server --routers random
```

上游提供 5 种路由器：`random` / `mf`（矩阵分解）/ `bert` / `causal_llm`（LLaMA-3-8B 分类器）/ `sw_ranking`（相似度加权 Elo）。

## 已知问题（待改造）

上游代码在生产视角下存在 6 类问题，详见 jobfinding 的改造方案文档：

1. 路由计算无缓存，每次请求重复调 Embedding API
2. 零可观测性，无 metrics / tracing / 结构化日志
3. 无容错机制，上游故障即全盘不可用
4. 同步阻塞，异步框架下慢请求阻塞事件循环
5. 无部署基建，纯手工启动、API Key 走命令行
6. 无网关能力，缺 `/v1/models` 端点

## 目录结构（上游原始）

```
routellm/
├── controller.py                路由控制器，封装 LiteLLM 调用
├── openai_server.py             FastAPI OpenAI 兼容服务
├── calibrate_threshold.py       阈值校准工具
├── routers/                     5 种路由器实现
│   ├── routers.py
│   ├── matrix_factorization/
│   ├── causal_llm/
│   └── similarity_weighted/
├── evals/                       评测框架 + 预计算响应数据（14M）
└── tests/                       2 个基础测试
examples/router_chat.py          Gradio 聊天界面
config.example.yaml              路由器配置
```

22 个 Python 文件，3006 行代码。

## 许可证

沿用上游 MIT License，见 `LICENSE`。
