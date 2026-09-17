# 实验记录：网关鉴权 + Dashboard 独立容器 + compose 部署

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **状态**：✅ 通过（端到端验证于真实容器）

## 需求

1. 网关绑 `0.0.0.0` 对外暴露（供 Hermes Agent 等客户端调用）
2. **Dashboard 拆为独立容器**，端口与网关差异化
3. 网关加 **API key 鉴权** —— 用法与普通模型 API 服务一致
4. 改用 **docker compose** 部署

## 最终架构

```
宿主机                    容器                        职责
─────────────────────────────────────────────────────────────
0.0.0.0:6060  ───────→  routellm-deploy :6060     路由网关（需 Bearer key）
0.0.0.0:8092  ───────→  routellm-dashboard :8080  监控面板（免鉴权）

共享：/mnt/data/wanghesong/routellm/metrics/metrics.db
      （网关读写，面板只读挂载）
```

容器内端口由各服务自定（互不冲突，不同网络命名空间），宿主机端口按实际情况映射。

## 实现

### 1. API key 鉴权（`routellm/monitoring/auth.py`）

客户端用法与普通模型 API 服务一致：

```bash
curl -H "Authorization: Bearer *** \
     -X POST http://<host>:6060/v1/chat/completions -d '{...}'
```

| 项 | 说明 |
|---|---|
| 配置 | `ROUTELLM_GATEWAY_API_KEY`（逗号分隔支持多 key 轮换） |
| 未配置时 | **不鉴权**（内网/回环部署的向后兼容） |
| 白名单 | `/health`（容器探针）、`/dashboard`、`/metrics`、`/api/*`（运维端点） |
| 401 响应 | 沿用 OpenAI 错误格式（客户端可正确解析） |
| 比较 | `hmac.compare_digest` 防时序侧信道 |

### 2. Dashboard 独立容器

- `Dockerfile.dashboard`：只装 FastAPI + uvicorn（**不含** litellm/torch/pandas）
- 只复制面板所需代码（store/metrics/dashboard），保持镜像精简
- 只读挂载 metrics 目录（避免与网关争 SQLite 写锁）

### 3. compose 编排

`docker-compose.yml` 定义两个服务 + 共享 metrics 目录 + healthcheck + 日志轮转。
`.env` 保存密钥（chmod 600，已加入 `.gitignore`）。

## 踩坑记录（5 个，均由端到端验证发现）

### 1. 【致命】鉴权中间件拦截了 lifespan scope

**症状**：容器健康检查通过，但**所有业务请求 500**：
```
AttributeError: 'NoneType' object has no attribute 'acompletion'
```
启动日志：
```
INFO: Waiting for application startup.
WARNING: auth: 鉴权失败: path= (key 缺失)     ← path 为空！
INFO: ASGI 'lifespan' protocol appears unsupported.
INFO: Application startup complete.
```

**根因**：`ApiKeyMiddleware.__call__` 没有判断 `scope["type"]`。lifespan scope
**没有 `path` 字段**，`requires_auth("")` 返回 True → lifespan 被 401 拦截 →
`lifespan` 里的 `CONTROLLER = Controller(...)` **永不执行** → 业务全部 500。

**修复**：显式透传非 HTTP scope：
```python
if scope.get("type") != "http":
    return await self.app(scope, receive, send)
```

**教训**：ASGI 中间件必须处理 lifespan/websocket scope。健康检查通过 ≠ 应用
初始化成功 —— healthcheck 只测 HTTP 端点，测不出 lifespan 是否跑完。

### 2. 【误导】uvicorn.run 传字符串 vs 传对象

`uvicorn.run("routellm.openai_server:app", workers=0)` 中：
- 字符串形式会让 uvicorn 重新 import 模块，理论上产生两份模块实例
- `workers=0` 会让 uvicorn 走多进程分支

**实测**：容器内确认**只有一个进程**（PID 1），所以这不是本次 500 的根因。
但仍改为传 **app 对象** + `workers=None` —— 消除隐患，语义更明确。
（排查过程中曾误判此点，记录以避免后人重复走弯路。）

### 3. docker 端口映射与容器内 bind 冲突

用 `-p 8091:8091` 同时让容器内进程 bind 8091 → Docker 的端口代理已占用
该端口 → 容器内 bind 失败。

**正解**：容器内端口与宿主机端口**不必相同**（`-p 8092:8080`），这样
容器内服务绑 8080、Docker 代理监听宿主机 8092，互不冲突。

### 4. 命名卷权限导致 SQLite 无法打开

```
sqlite3.OperationalError: unable to open database file
```
compose 命名卷默认 **root 拥有**，容器内非 root 用户无法写。

**修复**：改用宿主机目录挂载 + 预先 `chmod 777`。面板侧用 `:ro` 只读挂载。

### 5. 包初始化热切导入拖垮轻量容器

`monitoring/__init__.py` 里 `from .middleware import MetricsMiddleware` →
Dashboard 容器（无 litellm）import 本包时直接
`ModuleNotFoundError: No module named 'routellm.monitoring.middleware'`。

**修复**：`__init__.py` 不做热切导入，各子模块按路径显式导入。
由 `tests/test_monitoring_package_lazy.py` 守护该不变量。

### 6. 环境限制：Docker Hub 不可达

`docker pull python:3.11-slim` 超时；走 `docker.m.daocloud.io/library/python:3.11-slim`
成功（Prometheus 镜像同理）。

## 端到端验证（43，真实容器）

```
容器状态：
  routellm-dashboard  Up (healthy)  0.0.0.0:8092->8080/tcp
  routellm-deploy     Up (healthy)  0.0.0.0:6060->6060/tcp

鉴权：
  无 key        → 401  ✅
  错误 key      → 401  ✅
  正确 key      → 200  ✅
  /health       → 200（免鉴权）✅

业务（正确 key）：
  路由到 deepseek-ai/DeepSeek-V4-Flash（弱模型）
  内容 "1+1 equals 2."

Dashboard 容器（8092）：
  /dashboard → 200，10128B
  /health    → {"status":"online","db_ok":true,"total_requests":21}
  /api/metrics/summary → 成本节省 $0.085197，延迟分位正常

lifespan：
  "Application startup complete"（不再有 appears unsupported）
```

## 测试

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/test_gateway_auth.py` | 15 | key 解析、Bearer 提取、白名单、401 格式、中间件拦截/放行、**lifespan/websocket 透传** |
| `tests/test_gateway_auth_integration.py` | 5 | 网关 app 端到端鉴权（含"/health 免鉴权"） |
| `tests/test_dashboard_port.py` | 9 | 独立 app 路由、不含业务路由、端口配置、**面板启动失败返回 None** |
| `tests/test_monitoring_package_lazy.py` | 2 | 包初始化不拉起重依赖 |

全量回归：**133 passed, 17 skipped**。

## 部署方式

```bash
cd <repo>
# .env 需含（脚本已自动生成，权限 600）：
#   ROUTELLM_STRONG_MODEL / ROUTELLM_WEAK_MODEL / ROUTELLM_API_BASE
#   ROUTELLM_API_KEY          下游 LLM key
#   ROUTELLM_GATEWAY_API_KEY  本网关对外 key
#   ROUTELLM_METRICS_HOST_DIR metrics 宿主机目录

docker compose build
docker compose up -d
docker compose ps
```

客户端（如 Hermes Agent）配置：
```
base_url: http://<host>:6060/v1
api_key:  <ROUTELLM_GATEWAY_API_KEY>
model:    router-remote_bert-0.5
```

## 未做 / 已知限制

1. 面板 `/dashboard` **无鉴权**（绑 0.0.0.0）—— 依赖外层防火墙限制来源
2. 面板只读挂载 metrics，但 SQLite 跨容器共享在**高并发下仍有锁风险**
   （当前低流量安全；扩容时应改为从 Prometheus 读）
3. commit 时未把 `.env` 加入版本控制（有意为之，密钥不入库）
