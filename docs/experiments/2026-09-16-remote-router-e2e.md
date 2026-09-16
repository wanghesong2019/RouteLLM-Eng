# 实验记录：RouteLLM 与推理服务链路打通

- **日期**：2026-09-16
- **机器**：33 号机（RouteLLM 侧） + 43 号机（推理服务侧）
- **代码**：`routellm/routers/remote.py`（新增）+ `routellm/routers/routers.py`（注册）
- **脚本**：`scripts/test_e2e_remote_router.py`、`tests/`
- **状态**：✅ 通过

## 实验目的

把 `RemoteBERTRouter` 接入 RouteLLM，验证"RouteLLM → HTTP → 推理服务 → 路由决策"
整条链路真实可用（而非仅 mock 通过）。

## 实现

### `routellm/routers/remote.py`

新增 `RemoteBERTRouter`，实现上游 `Router` 抽象接口：

```python
class RemoteBERTRouter(Router):
    NO_PARALLEL = True

    def __init__(self, base_url, timeout=30.0, batch_size=64, model_type="bert"): ...
    def calculate_strong_win_rate(self, prompt) -> float: ...          # Router 契约
    def calculate_strong_win_rate_batch(self, prompts) -> list[float]: ...  # 批量
    def health(self) -> dict: ...                                      # 运维
```

设计要点：

- **单条调用复用批量路径**（`calculate_strong_win_rate` → `_batch([prompt])[0]`），避免两套代码
- **分片**：超长列表按 `batch_size` 拆分，顺序保持
- **响应校验**：结构不符 / 数量不匹配 / 缺字段 → 抛 `RemoteInferenceError`
- **不依赖 `routellm` 上游模型加载逻辑**，纯 HTTP

### 注册

`routers.py` 中延迟导入并注册（避免循环依赖）：

```python
ROUTER_CLS = {..., "remote_bert": _RemoteBERTRouter}
```

## TDD 过程

### RED

先写测试（`tests/test_remote_bert_router.py`、`tests/test_router_registration.py`），
确认失败：

```
12 failed — ModuleNotFoundError: No module named 'routellm.routers.remote'
```

### GREEN

实现后：

```
19 passed（12 远程路由器 + 7 注册与集成）
```

### 过程中的两处修正（均为测试侧问题，非实现缺陷）

1. **`model_counts` 断言错误** —— 上游 `Controller.route()` **不记录**计数，
   只有 `_get_routed_model_for_completion()`（即 completion 路径）记录。
   修正为走 completion 路径验证。

2. **暴露上游依赖缺陷** —— `batch_calculate_win_rate` 的分支逻辑：

   ```python
   if router_instance.NO_PARALLEL and self.progress_bar:
       return prompts.progress_apply(...)
   elif router_instance.NO_PARALLEL:
       return prompts.apply(...)
   else:
       return prompts.parallel_apply(...)   # ← 需要 pandarallel
   ```

   `pandarallel` 仅在 `eval` optional-dependencies 中，**非核心依赖**。
   任何 `NO_PARALLEL = False` 的路由器（如上游 `random`）走评测路径都会
   `AttributeError: 'Series' object has no attribute 'parallel_apply'`。

   → `RemoteBERTRouter` 设 `NO_PARALLEL = True`：HTTP 调用场景下，
   pandarallel 的进程池开销大于收益，且避免引入额外依赖。

## 端到端验证

### 网络前提：43 防火墙限制

**重要发现**：43 号机仅开放 SSH 端口（20007/20031），**6070 从外部不可达**。

```
33 → 43 端口探测:
   22    filtered
   20007 OPEN       (SSH)
   20031 OPEN       (SSH)
   6070  filtered   ← 推理服务
   8090  filtered
```

因此本次验证通过 **SSH 隧道**：

```bash
ssh -N -L 16070:127.0.0.1:6070 43
```

### 验证结果

```
1. 连通性
   status=online  model_loaded=True
   arch=xlm-roberta  device=cuda  labels=3  load_seconds=1.12

2. win_rate 一致性（经 SSH 隧道，跨机器）
   prompt                          service  baseline   匹配
   What is 1+1?                     0.2970    0.2970    ✓
   hi                               0.4007    0.4007    ✓
   Write a poem about the sea.      0.1993    0.1993    ✓

3. 路由决策（Controller + remote_bert, threshold=0.5）
   What is 1+1?                     0.2970 → weak
   Write a poem about the sea.      0.1993 → weak
   Prove √2 is irrational...        0.4769 → weak

4. 批量（100 条经隧道）
   114.2ms（1.14ms/条）
   win_rate 范围: 0.4369 ~ 0.4751

5. 错误处理
   指向不存在端口 → RemoteInferenceError ✓
```

**结论**：跨机器链路无数值偏差（三例逐位一致），路由决策正确，
批量与错误处理均符合预期。

## 对 Docker 化的影响（重要）

43 的防火墙策略决定了部署位置：

| 方案 | 可行性 |
|---|---|
| RouteLLM 容器在 33，推理服务在 43 | ✗ 网络隔离，需 SSH 隧道（容器内不便） |
| RouteLLM 容器在 33 + 隧道 sidecar | ⚠️ 可行但复杂 |
| **RouteLLM 容器与推理服务同在 43** | ✅ **推荐** —— 容器经 `host.docker.internal:6070` 访问 host 服务 |

**决策**：Docker 化时两者都部署在 43。推理服务在 host 运行（模型权重留 host），
RouteLLM 容器通过 `host.docker.internal` 访问 —— 与 ADR-001 的架构一致。

## 复现方式

```bash
# 1. 43 上启动推理服务（见 services/README.md）
# 2. 建立隧道
ssh -N -L 16070:127.0.0.1:6070 43 &
# 3. 跑端到端验证
ROUTELLM_INFERENCE_URL=http://127.0.0.1:16070 \
    python scripts/test_e2e_remote_router.py
# 4. 跑单元测试
pytest tests/ -v
```

## 遗留事项

1. **43 防火墙**：若需长期从 33 访问推理服务，应固化隧道（autossh / systemd）
   或与运维确认是否可为特定网段放行 6070。当前用临时隧道。
2. **CausalLLM 未接入**：`RemoteBERTRouter` 已参数化 `model_type`，
   待 CausalLLM 实现后可复用同一客户端（或抽出 `RemoteRouterBase`）。
