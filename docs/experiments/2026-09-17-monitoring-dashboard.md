# 实验记录：监控体系 + Dashboard（P0）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **模块**：`routellm/monitoring/`（metrics / store / prometheus / middleware / dashboard）
- **状态**：✅ 通过（端到端验证于 6060 真实网关）

## 目标（方案文档 4.2 节，P0）

改造前 `openai_server.py` 的状况（文档 4.2.1 的问题定位）：

```python
count = defaultdict(lambda: defaultdict(int))   # 全局变量，重启丢失
logging.info(CONTROLLER.model_counts)           # 只打印，无结构化
return JSONResponse({"status": "online"})       # /health 什么都看不到
```

**缺失的指标**：路由决策、置信度（win_rate）、延迟分位、成本节省、
缓存命中率、错误率。

## 方案（文档 4.2.3：方案 B + A 混合）

| 组成 | 说明 |
|---|---|
| 自研 Dashboard | FastAPI + SQLite + 单页 HTML/ECharts，展示全栈能力 |
| Prometheus 端点 | 兼容标准监控生态 |

**为何自研而非 Grafana**：RouteLLM 的核心价值是「省钱」，自研面板可直接
展示成本节省曲线，比配置 Grafana 面板直观，且无需额外部署组件。

## 实现

```
routellm/monitoring/
├── metrics.py      指标模型 + 成本估算 + prompt hash
├── store.py        SQLite 持久化（解决「重启丢失」）
├── prometheus.py   Prometheus 文本格式（零依赖自实现）
├── middleware.py   FastAPI 中间件（请求级采集）
└── dashboard/
    ├── __init__.py 路由：/dashboard + /api/metrics/*
    └── static/index.html  单页 Dashboard（ECharts）
```

### 关键设计决策

| 决策 | 理由 |
|---|---|
| SQLite 而非内存 | 原问题就是「重启丢失」，必须持久化 |
| Prometheus 自实现而非用 prometheus_client | 保持网关轻量（ADR-001）；文本暴露格式简单，几行字符串 |
| 指标只存 prompt **hash** | 隐私；且与缓存 key 同算法，便于关联 |
| 中间件只拦 `/v1/chat/completions` | 运维端点不产生路由成本，记录会污染统计 |
| 写入用 `asyncio.create_task` | fire-and-forget，不阻塞响应 |
| 路由元信息用 **ContextVar** | 见下节「踩坑」 |
| 成本在 store 侧重算，不信任传入值 | 见下节「踩坑」 |

## 踩坑记录（三个，均由验证发现）

### 1. Starlette 中间件参数顺序

`app.add_middleware(cls, store=..., ctx_var=...)` 内部以
**`cls(app, *args, **kwargs)`** 构造中间件。因此中间件签名的**第一个
位置参数必须是 `app`**：

```python
# ✗ 错误：TypeError: got multiple values for argument 'store'
def __init__(self, store, app, ctx_var=None): ...

# ✓ 正确
def __init__(self, app, store=None, ctx_var=None): ...
```

### 2. SQLite 内存库必须设 row_factory

`:memory:` 连接不设 `row_factory = sqlite3.Row` 时，`fetchone()` 返回
**tuple**，后续 `row["total"]` 会 `TypeError: tuple indices must be
integers`。文件库因在 `_conn()` 里设了而正常 —— 只有内存库踩到。

### 3. 成本口径不一致导致「节省额」被截断为 0

初版 `summary()` 用**入库时传入的 `estimated_cost`** 作为实际成本，
但「若全走强模型」是按 **token 数 × 单价**算的。两者口径不同 →
`max(0, if_strong - actual)` 被截成 0，节省额恒为 0。

**修正**：实际成本也由 store 按 `routed_model + token 数` 重算，
不信任传入值。

### 4. 路由元信息曾丢失（端到端验证发现）

首版部署后 Dashboard 显示 `win_rate = N/A`、`routing_latency_ms = 0`。
原因：`Controller._get_routed_model_for_completion` 调 `route()` 只拿模型名，
**把 win_rate 和耗时都丢了**。

**修正**：改为显式调 `calculate_strong_win_rate()` 拿到置信度，
记录到 `RoutingInfo`（存在模块级 ContextVar 中，按协程隔离，
避免并发请求串台）。网关侧再读取。

## 端到端验证（43，6060 真实网关）

打入 4 个难度不同的真实请求后：

```
=== /api/metrics/summary ===
total_requests: 3
strong_count: 0, weak_count: 3
actual_cost_usd: 9.9e-05
cost_if_strong_usd: 0.01115
cost_saved_usd: 0.011051          ← 成本节省可算
latency_ms: p50 2628.8, mean 5333.19
```

修复路由元信息后：

```
=== /api/metrics/recent ===
remote_bert  weak  win_rate=0.4770  routing=28.5ms  cost=0.000262   ← 证明√2（最难）
remote_bert  weak  win_rate=0.3318  routing=28.6ms  cost=0.000255
remote_bert  weak  win_rate=0.3646  routing=26.0ms  cost=0.000016   ← haiku
remote_bert  weak  win_rate=0.4007  routing=36.5ms  cost=0.000037   ← 法国首都
```

**win_rate 区分度正确**（0.33~0.48，难题最高），路由延迟真实采集（26~36ms）。
（均路由到 weak 因阈值 0.5 且 win_rate < 0.5，符合逻辑。）

### 端点验证

| 端点 | 结果 |
|---|---|
| `GET /dashboard` | HTTP 200，10128 字节 HTML |
| `GET /metrics` | Prometheus 文本格式（counter + histogram 含 bucket/sum/bar） |
| `GET /api/metrics/summary` | 聚合指标 JSON |
| `GET /api/metrics/recent` | 最近请求列表 |
| `GET /api/metrics/timeseries` | 时间桶序列 |
| `GET /health` | 含 metrics 摘要 |

### 数据持久化

metrics DB 挂载到 host：`/mnt/data/wanghesong/routellm/metrics/metrics.db`
→ 容器 `/data/metrics.db`。**重启容器不丢指标**（对比原实现的内存 dict）。

## 测试

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/test_monitoring.py` | 18 | 模型字段、成本估算、SQLite 存取/分位/占比/节省/命中率/错误率/时序/空库、Prometheus counter/gauge/histogram/渲染、Dashboard 路由/API/静态文件 |
| `tests/test_controller_metrics.py` | 5 | RoutingInfo 字段、记录/读取、ContextVar 隔离、覆盖语义 |

全量回归：**102 passed, 17 skipped**。

## Dashboard 面板

```
┌──────────────────────────────────────────────────────────┐
│ RouteLLM Dashboard                             [刷新]    │
├─────────┬─────────┬─────────┬─────────┬─────────┬────────┤
│ 总请求数 │ 强模型占比│ 成本节省 │ 平均延迟 │ 路由延迟 │ 缓存命中│
├─────────┴─────────┴─────────┴─────────┴─────────┴────────┤
│ [请求量与强模型占比 - 折线]   │ [强/弱模型分配 - 饼图]      │
│ [延迟分布 P50/P95/P99 - 柱]   │ [成本累积 实际vs全强 - 柱]  │
│ [最近请求列表 - 表格（含 win_rate / cached 标记）]         │
└──────────────────────────────────────────────────────────┘
```

前端：单页 HTML + ECharts（CDN），**10 秒自动刷新**，深色主题。

## 未做的事

1. **告警规则未配置** —— Prometheus 端点已就绪，但未接 Alertmanager
2. **前端未做鉴权** —— `/dashboard` 无认证（内网部署可接受，公网需加）
3. **ECharts 走 CDN** —— 离线环境下图表不显示（页面本身仍可用）
4. **时序数据未做降采样/清理** —— SQLite 会持续增长，缺 TTL 清理

## 复现方式

```bash
# 网关启动时自动启用（默认开启）
export ROUTELLM_METRICS_ENABLED=1
export ROUTELLM_METRICS_DB=/data/metrics.db
# 启动后访问：
#   http://<host>:6060/dashboard          面板
#   http://<host>:6060/metrics            Prometheus
#   http://<host>:6060/api/metrics/summary   API

# 关闭监控
export ROUTELLM_METRICS_ENABLED=0

# 跑测试
pytest tests/test_monitoring.py tests/test_controller_metrics.py -v
```
