# 实验记录：运行时配置热更新（Phase 3.5）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090）
- **模块**：`routellm/config_runtime/`（store / api）+ Dashboard 转发层 + 配置页
- **状态**：✅ 通过（端到端验证于真实容器）

## 需求（方案文档 4.8 节 / 问题7）

下游强弱模型的 `base_url` / `api_key` / 模型名可**在不重启网关**的前提下
编辑并立即生效；前端在 Dashboard 上提供编辑口子。

**改造前的痛点**：这些配置通过环境变量在容器启动时注入
（`Settings.from_env()`），改任何一项都要 `docker compose up -d` 重建容器
（服务中断 ~30s）。

## 架构（方案 B —— 配置存网关，Dashboard 只转发）

```
浏览器
  │  （只与面板通信，不接触网关 key）
  ▼
Dashboard 容器 (:8092)  /config 页面 + /api/config 转发
  │  HTTP + Bearer key（容器网络内 http://routellm:6060）
  ▼
网关容器 (:6060)  /api/config  ← 配置真相（single source of truth）
  │
  ▼
Controller.live_config()  ← 按请求无锁读取 → 立即生效
  │
  ▼
RuntimeConfigStore  →  持久化到共享卷 /data/runtime_config.json (600)
```

**为何选方案 B**（对应方案文档 4.8.2 的三方案对比）：
1. 配置归网关所有，面板不持有配置真相 —— 职责清晰
2. 保持面板「只读挂载 metrics」的干净边界，不给它配置写权限
3. **前端不需要知道网关 key** —— key 只存在于面板容器的环境变量里，
   这比让浏览器直连网关（方案C）更安全
4. 转发层可集中做错误归一化（网关不可达 / 401 / 5xx 都转成前端可展示的结构）

## 实现要点

### 1. 不可变配置 + 原子替换（`config_runtime/store.py`）

```python
@dataclass(frozen=True)
class RuntimeConfig:
    strong_model: str; weak_model: str; api_base: str; api_key: str

def update(self, **fields):
    new_cfg = replace(self._config, **fields)   # 构造新对象
    self._config = new_cfg                      # 原子替换引用
```

**为何这样设计**：请求处理时读到的一定是某个完整版本，绝不会读到
"半新半旧"的组合。CPython 中对象引用赋值是原子的，因此**读取方无需加锁**
—— 请求路径 O(1)、零开销。这是选「不可变+替换」而非「可变+加锁」的
关键收益。

### 2. 持久化的原子写

用 `tempfile` + `os.replace` —— 避免进程在写盘中途被杀导致配置文件损坏。

### 3. Controller 按请求现取

```python
# 改造前：__init__ 时固化
return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)

# 改造后：按请求现取
kw = self.downstream_kwargs(kwargs["model"])   # 从 live_config() 取
return await acompletion(**kw, **kwargs)
```

未注入 `config_store` 时回落构造参数 —— **向后兼容，行为不变**。

### 4. 安全设计

| 措施 | 说明 |
|---|---|
| 掩码回显 | GET 返回 `sk-***7890`，明文不出后端 |
| 面板不持有 key | 前端只与面板通信；key 在面板容器环境变量 |
| 鉴权 | 复用 `ApiKeyMiddleware`；配置 API 额外做二次校验（不依赖路径白名单） |
| 审计日志 | 记录变更字段名与时间，**不记录值**（值可能含密钥） |
| 文件权限 | 600 |
| 连通性预检 | 可选：改 base_url 前试探下游，失败则拒绝写入 |

### 5. 前端（`static/config.html`）

- 独立页面 `/config`（与监控展示分开 —— 两种不同操作场景）
- 只提交**有变化**的字段；api_key 留空表示不修改（避免误清空）
- 提供「测试连接」「保存并立即生效」「重新加载」
- 只读项明确标注（路由器类型、阈值、路由模型路径需重启）

## 端到端验证（43，真实容器）

```
=== 配置读取（掩码）===
{"strong_model":"openai/deepseek-ai/DeepSeek-V4-Pro",
 "weak_model":"openai/deepseek-ai/DeepSeek-V4-Flash",
 "api_base":"https://api.siliconflow.cn/v1",
 "api_key":"***","api_key_masked":true, ...}

=== 通过 Dashboard 转发更新 ===
  ok: True       changed: ['weak_model']

=== 网关侧确认（未重启）===
  weak: openai/deepseek-ai/DeepSeek-V4-Flash-HOTUPDATED    ← 立即生效

=== 容器启动时间（证明未重启）===
  网关: 2026-09-17T07:12:09Z   ← 未变

=== 真实请求验证 ===
  路由到 deepseek-ai/DeepSeek-V4-Flash，正常返回

=== 持久化 ===
  /data/runtime_config.json 内容正确，权限 -rw------- (600)
```

## 测试（TDD，37 例新增）

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/test_config_runtime.py` | 16 | 不可变性、原子替换、**并发一致性**（300 次更新 + 并发读，无半新半旧）、持久化、权限 600、损坏回落、掩码 |
| `tests/test_config_api.py` | 14 | GET 掩码、PUT 局部更新、空 payload 不清空、审计不含密钥、未知字段拒绝、连通性预检拦截坏配置 |
| `tests/test_controller_hot_reload.py` | 7 | **改配置无需重启即生效**、live_config/live_model_pair、向后兼容 |
| `tests/test_dashboard_config_proxy.py` | 13 | 转发 GET/PUT/verify、网关不可达/401/5xx 归一化、路由注册 |

全量回归：**183 passed, 17 skipped**。

## 踩坑记录

### 1. Dashboard 容器缺 httpx

转发层用了 `httpx`，但 `Dockerfile.dashboard` 的依赖清单没有它 →
运行时报 `ModuleNotFoundError: No module named 'httpx'`，被转发层的
错误处理转成了 `proxy_error`（**面板没崩，这是降级设计生效**）。

**修复**：Dockerfile.dashboard 加 `httpx>=0.27`。

### 2. 路由存在性检查被 FastAPI 包装对象误导

`app.include_router()` 后，`app.routes` 里出现的是 `_IncludedRouter`
包装对象（无 `.path`），直接 `any(r.path == "/api/config")` 会得到
**假阴性**。需展开 `r.original_router.routes` 才能查到。

（同一坑在 Phase 2 的 dashboard 测试里也踩过，已沉淀进
`tests/test_dashboard_port.py` 的 `_all_paths` 辅助函数。）

### 3. 验证脚本的引号层级

多层嵌套引号（bash → curl → python -c）极易被吃掉导致命令退化
（实测 `-X PUT` 被吞后变成 POST，服务端返回 405 而非 404，误导排查方向）。

**做法**：用 python 拼接生成脚本，避免手写嵌套引号；脚本内用
`$GW`/`$AUTH` 变量减少引号层数。

## 未做 / 演进项

1. **多副本一致性**：当前单容器部署无此问题；水平扩容时需改为从配置中心
   （etcd/Consul）广播
2. **配置版本历史/回滚**：当前只保留当前值，无历史版本与一键回滚
3. **加密存储**：容器内无 KMS，自实现加密收益有限；当前用「600 权限 +
   掩码回显 + 日志脱敏」。多租户环境应引入外部密钥管理
4. **热更范围**：仅 base_url / api_key / 模型名。路由器类型、阈值、
   路由模型路径仍需重启（UI 已标注为只读）

## 复现方式

```bash
# 网关侧（配置真相）
curl -H "Authorization: Bearer *** \
     http://<host>:6060/api/config

# 面板侧（转发，前端用这个）
curl http://<host>:8092/api/config
curl -X PUT http://<host>:8092/api/config \
     -H 'Content-Type: application/json' \
     -d '{"weak_model":"新模型名"}'

# 连通性预检
curl -X POST http://<host>:8092/api/config/verify

# 浏览器
#   http://<host>:8092/config     配置编辑页
#   http://<host>:8092/dashboard  监控面板（右上角有「运行时配置」入口）
```
