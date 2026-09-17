# 实验记录：强弱分侧配置 + 原始模型名 + 分侧连通性预检（Phase 3.6）

- **日期**：2026-09-17
- **机器**：43 号机（4×RTX 4090），真实容器验证
- **模块**：`routellm/config_runtime/`（store / api）+ `controller.py`
  + Dashboard 转发层 + 配置页
- **状态**：✅ 通过（端到端验证于真实容器）

## 起因：用户在 Dashboard 上发现的三个问题

上一轮 Phase 3.5（运行时配置热更新）完成后，用户在实际页面上发现：

1. **强弱配置不对等**：强模型有「模型名 / API Base URL / API Key」三项，
   弱模型只有「模型名」—— base_url / api_key 是网关级全局项，强弱共用。
   用户诉求：两者都要完整配置，且**各自配置对各自生效**。
2. **模型名必须带 `openai/` 前缀**：用户希望配置**原始模型名**
   （如 `deepseek-ai/DeepSeek-V4-Pro`），前缀不应由人手工写。
3. **「测试连接」只有一个按钮**：测的是哪个模型？界面无从知晓。

第 3 点查证属实且是设计疏漏：`verify` 内部只用 `strong_model` 做预检，
**从不碰 weak_model**；而前端 `setStatus` 只打印 `detail`，没回显
probed 的是谁 —— 用户看不出测的是强模型。

用户另外补充了第 4 点（重要的边界条件）：

4. **只填一个模型名时应可用**：没填的那一档不能报错，
   所有请求都走已填的那档。

## 设计（用户确认的规则）

### 模型层兜底：谁配用谁

    只配强 → strong = weak = 强模型名
    只配弱 → strong = weak = 弱模型名
    两侧都配 → 保持各自
    两侧都空 → **校验期报错**（fail-fast，不留到请求期）

为何不报错而是并入：用户可能只想先配一个模型跑通链路。
此时若"路由到空模型名"，下游会拿到 400 —— 是必须避免的失败模式。
两侧都空则是真错误，必须显式失败。

### 地址层回落：顶层作全局兜底

    strong_api_base or api_base       # 该侧为空 → 回落顶层
    weak_api_key    or api_key

**为何保留顶层**：向后兼容零破坏 —— 旧配置（只有顶层 api_base/api_key）
读进来后每侧字段为空 → 全部回落顶层 → 行为与改造前完全一致。
"只填顶层一份"= 强弱共用（旧用法）；"两侧各填"= 各自生效。

### 模型名前缀：存储原始名，调用层拼接（方案 a）

用户选定方案 a（小而稳）：配置层与前端都存/显示**原始名**，
由 `Controller.downstream_kwargs` 在调用前的**最后一步**统一拼 `openai/`。

为何必须拼：litellm 靠前缀选择适配器，去掉会抛 BadRequestError。
所以拼接必须只此一处，不能散落在前端或 store。

已带前缀的名字原样返回（避免 `openai/openai/...`）。识别用**白名单**
而非"含斜杠即视为前缀"—— 模型名本身常含斜杠
（`deepseek-ai/DeepSeek-V4-Pro`），后者会误判。

## 顺带发现的既有缺陷：转发层把 4xx 包成 200

验证时发现：非法 tier 明明被网关拒绝（422），面板对前端却报 **HTTP 200**。

    POST /api/config/verify?tier=medium
      网关 → 422 {"detail": "未知 tier: 'medium'"}
      面板 → 200 {"proxy_error": "网关返回 HTTP 422", ...}   ← 前端误判为成功

根因：转发层把**所有** `>=400` 统一转成 `200 + proxy_error`。
归一化的初衷（面板不因网关异常整页崩）对 5xx 成立，
但用于 4xx 客户端错误就错了 —— 抹掉状态码后前端无法区分
"成功"与"参数错/鉴权失败"。

**修正**（独立于用户三个问题的第四个改动，经用户确认后实施）：

    4xx 客户端错误 → 抛 HTTPException，**保留状态码**，detail 原样透传
    5xx 服务端错误 → 归一化成 200 + proxy_error（既有降级设计不变）
    网关不可达     → 归一化成 200 + proxy_error（同上）

前端相应加 `if (!r.ok)` 分支（load / save / verify 三处）。

## 实现要点

### 1. 配置模型扩展（`store.py`）

```python
@dataclass(frozen=True)
class RuntimeConfig:
    strong_model: str; weak_model: str
    api_base: str; api_key: str                      # 顶层 = 全局兜底
    strong_api_base: str = ""; strong_api_key: str = ""
    weak_api_base: str = "";   weak_api_key: str = ""
```

新增方法：

| 方法 | 作用 |
|---|---|
| `effective_api_base(tier)` / `effective_api_key(tier)` | 该侧优先，空回落顶层；非法 tier 抛 ValueError |
| `effective_model_pair()` | 含单模型兜底，返回 `ModelPairView(strong, weak)` |

`ModelPairView` 独立于 `controller.ModelPair` —— 避免 config 层反向依赖
controller 层，保持 `config_runtime` 可单独导入与测试。

### 2. 向后兼容的配置加载

`_load_initial` 改为按字段名列表取值，缺失字段按 `""` 处理：

```python
cfg = RuntimeConfig(**{k: data.get(k, "") or "" for k in self._FIELDS})
```

旧配置文件（无分侧字段）→ 每侧为空 → 回落顶层 → 行为不变。

### 3. Controller 按 tier 取配置

```python
kw = self.downstream_kwargs(model, tier="strong")   # 用强侧 base/key
kw = self.downstream_kwargs("deepseek-ai/X")        # tier=None → 顶层（兼容）
```

`live_model_pair()` 在注入 store 时走 `effective_model_pair()`，
因此单模型兜底对**路由决策**同样生效（不会返回空模型名）。

### 4. 分侧连通性预检

`POST /api/config/verify?tier=strong|weak`，响应明示被测对象：

```json
{"ok": true, "tier": "weak",
 "probed": {"tier": "weak", "model": "...", "api_base": "...",
            "api_key": "sk-***nzfp", "source": "global"},
 "note": "仅验证地址可达与密钥有效；模型名可用性需实际调用才可确认"}
```

`source` 说明凭据来源（该侧覆盖 `weak` / 顶层兜底 `global`），
前端渲染成标签，用户一眼看出"这个按钮测的是谁、用的哪套凭据"。

**能力边界写进响应**：`/models` 只验证地址与密钥，
**不能**证明具体模型名可用 —— 多数 OpenAI 兼容网关不校验 `/models`
可见性。UI 文案明确这一点，避免误解为"模型可用性已验证"。

### 5. 前端（`static/config.html`）

- 强 / 弱各一个 fieldset，各有完整三项（模型名 / Base URL / Key）
- **两个独立的「测试强模型连接」「测试弱模型连接」按钮**，各带独立状态行
- 全局兜底单独一个 fieldset，标注"上面两侧留空时使用这里"
- 单模型模式时顶部黄色横幅提示"所有请求都会走 X"
- 密钥输入框留空 = 不修改（掩码作 placeholder）

## 端到端验证（43，真实容器）

```
[0] 初始配置：分侧字段为空，single_model_mode=false   ← 旧配置兼容

[1] 分侧字段可编辑
    OK  strong_api_base / strong_api_key / weak_api_base / weak_api_key

[2] 模型名以原始名存储
    PUT strong_model=test-org/Test-Model-Raw  → 回显 'test-org/Test-Model-Raw'
    （无 openai/ 前缀）

[3] 强弱各自配置（隔离性）
    strong_api_base = 'https://strong.example.com/v1'
    weak_api_base   = 'https://weak.example.com/v1'

[4] 两个「测试连接」各测各的
    [tier=strong] 测的模型='test-org/Test-Model-Raw'  凭据来源='strong'
    [tier=weak]   测的模型='...DeepSeek-V4-Flash'      凭据来源='weak'
    OK  非法 tier=medium -> HTTP 422
        detail = 未知 tier: 'medium'（应为 ('strong', 'weak') 之一）

[5] 单模型兜底
    PUT weak_model=''  → single_model_mode=True
    effective_strong = effective_weak = 'test-org/Test-Model-Raw'（非空）

[7] 恢复原配置 → 成功
```

容器内直接验证核心逻辑（绕过网络依赖）：

```
单模型模式（只配 weak）: strong = weak = 'only-one-model'
前缀拼接: deepseek-ai/DeepSeek-V4-Pro → openai/deepseek-ai/DeepSeek-V4-Pro
          openai/deepseek-ai/DeepSeek-V4-Pro → 原样（不重复加）
          qwen-max → openai/qwen-max
```

## 测试（TDD，新增 43 例）

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/test_config_per_tier.py` | 17 | 字段扩展、地址回落、隔离性、单模型兜底、两侧皆空报错、前缀拼接、掩码 |
| `tests/test_config_per_tier_api.py` | 13 | 可编辑字段、分侧 PUT、空串清空 vs None 不改、分侧 verify、source 回显、非法 tier 422 |
| `tests/test_proxy_error_semantics.py` | 10 | 4xx 保留状态码、5xx/不可达仍归一化、成功路径不变 |
| `tests/test_dashboard_config_proxy.py` | +3 | verify 的 tier 透传 |
| `tests/test_dashboard_config_proxy.py` | 1 改 | `test_gateway_401_propagates` 更新为新约定（状态码透传） |

全量回归：**238 passed, 17 skipped**。

### 反向验证（确认测试真能抓错）

写完实现后故意破坏，确认测试变红，再恢复：

| 破坏项 | 结果 |
|---|---|
| verify 忽略 tier（永远测全局） | 1 failed（凭据串了） |
| 去掉单模型兜底（回到"空就是空"） | 4 failed |
| `EDITABLE_FIELDS` 去掉分侧字段 | 3 failed |

## 踩坑记录

1. **43 的 `rag-dev` 环境没有 litellm** —— 该环境是给 bge-m3/torch 用的，
   `import routellm.controller` 直接 ModuleNotFoundError。
   Controller 相关测试在 43 上**只能走容器**（镜像内有 litellm）。
   非 Controller 的纯 config 测试可在宿主机跑。

2. **面板容器内访问自身要用容器端口 8080**，不是宿主机映射端口 8092；
   访问网关容器用**服务名** `routellm:6060`，不是 127.0.0.1。

3. **模型名格式易错**：`router-<router名>-<阈值>`，且路由器名是
   `remote_bert`（带下划线），完整 ID 为 `router-remote_bert-0.5`。
   写错会得到 400（`Invalid router bert`）或 500
   （`_parse_model_name` 解包失败），与配置功能无关，排查时别被带偏。

4. **litellm 启动拉模型价格表**会超时重试（本机网络访问
   raw.githubusercontent.com 不通），容器启动约需 40-50 秒才 healthy。
   等健康检查，别急着判定启动失败。

## 未做 / 演进项

1. **多副本一致性**：单容器部署无此问题；水平扩容需配置中心广播
2. **配置版本历史 / 回滚**：当前只保留当前值
3. **加密存储**：容器内无 KMS；当前用「600 权限 + 掩码回显 + 日志脱敏」
4. **模型可用性预检**：当前 `/models` 只验证地址与密钥。
   若要真验证模型名可用，需发一次极小的 `chat/completions`（有成本，
   且可能触发计费）—— 列为可选增强
5. **热更范围**：仅 base_url / api_key / 模型名。路由器类型、阈值、
   路由模型路径仍需重启（UI 已标注只读）

## 复现方式

```bash
# 面板链路（容器内执行用 8080，宿主机用 8092）
GW=http://127.0.0.1:8080 ROUTELLM_GATEWAY_API_KEY=<key> \
    python3 scripts/verify_per_tier_config.py

# 浏览器
#   http://<host>:8092/config     强弱两套完整配置 + 两个独立测试按钮
#   http://<host>:8092/dashboard  监控面板
```
