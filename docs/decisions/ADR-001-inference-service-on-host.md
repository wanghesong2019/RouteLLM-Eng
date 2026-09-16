# ADR-001：推理模型下沉至 host 独立服务

- **状态**：已采纳
- **日期**：2026-09-16
- **决策者**：项目负责人 + 开发

## 背景

上游 RouteLLM 的 `BERTRouter` 与 `CausalLLMRouter` 在**进程内**用 transformers 加载模型：

```python
# routellm/routers/routers.py
class BERTRouter(Router):
    def __init__(self, checkpoint_path, num_labels=3):
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, ...)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
```

本项目的目标是 Docker 化部署。若维持进程内加载，容器需要同时包含：

1. CUDA 版 torch（约 2-3GB）
2. 模型权重：BERT/xlm-roberta 1.1GB + CausalLLM（LLaMA-3-8B）约 17GB
3. transformers / datasets 全家桶

这会导致镜像臃肿、构建缓慢、且每次改代码重建镜像都要重新打包权重。

同时 RouteLLM 还需保留 `sw_ranking` / `mf` 两个路由器，它们依赖 numpy + `datasets.load_dataset()` + OpenAI Embedding API，无法（也不应）下沉。

## 决策

**将 BERT / CausalLLM 的模型推理下沉到 host（43 号机）上的独立 FastAPI 服务，RouteLLM 容器通过 HTTP 调用。**

具体形态：

```
┌─────────────────────────────┐
│  RouteLLM 容器               │
│  - sw_ranking / mf / random  │
│  - 路由决策 + 缓存 + 监控     │
│  - CPU torch（如需）         │
└──────────┬──────────────────┘
           │ HTTP (win_rate)
           ▼
┌─────────────────────────────┐
│  host:43 推理服务            │
│  - BERT (xlm-roberta)        │
│  - CausalLLM (LLaMA-3-8B)    │
│  - transformers + CUDA torch │
│  - 模型权重留在 host         │
└─────────────────────────────┘
```

推理服务**独立实现**，不 import `routellm` 包，只依赖 transformers + fastapi。

## 排除的备选方案

### 方案 A（排除）：全部塞进容器

模型权重 + CUDA torch 打进镜像。

排除理由：
- 镜像体积 20GB+
- 43 号机根分区剩 90G（虽然 `/mnt/data` 有 2.6T，但 Docker 默认数据目录在根分区）
- 每次改代码重建镜像需重新打包权重，迭代慢

### 方案 B（排除）：权重挂载进容器

用 volume 挂载权重目录，容器内 torch 直接加载。

排除理由：
- 容器仍需 CUDA torch + transformers（约 3GB）
- 容器需要 GPU 透传（`--gpus`），增加部署复杂度
- 与"推理服务可独立扩缩容"的目标不符

### 方案 C（采纳）：推理 HTTP 服务化

即本文档决策。

### 方案 D（评估后未采用）：vLLM 承载推理

用 vLLM 起推理服务替代 transformers。

评估结论：**不适用**。原因：

1. `routellm/causal_llm_gpt4_augmented` 不是标准 causal LM —— 它在词表末尾追加了 5 个特殊 token `[[1]]`~`[[5]]`（见模型文件 `new_embeddings.safetensors`），推理方式是从这 5 个 token 的 logits 反推 win rate，而非普通生成。
2. 该场景每次只生成 1 个 token，vLLM 的吞吐优势无法体现。
3. 上游 `CausalLLMClassifier` 持有自定义模型对象，vLLM 未必兼容。

实测数据支撑：vLLM 0.8.5 安装在 43 号机 `reflexicoder` 环境（已验证可用），但 `causal_llm` 模型在 hf-mirror 上仅 109 次下载（对比 bert 的 41472 次），兼容性风险高。

## 影响

### 正面

- RouteLLM 容器轻量，无需 CUDA 与权重
- 推理服务可独立重启 / 扩容，不牵连网关
- 容器化与模型部署解耦，各自迭代
- 两个模型共享一个 GPU 服务，避免重复加载

### 负面 / 需处理

- **新增网络跳数**：路由决策增加一次 HTTP 往返（本机环回，预计 <1ms，待实测）
- **新增运维面**：推理服务需单独管理（启动、健康检查、日志）
- **服务发现**：容器需知道 host 地址 —— Docker 下用 `host.docker.internal` 或 `network_mode: host`
- **一致性要求**：推理服务的行为必须与上游 `calculate_strong_win_rate` 完全一致，需测试保障

## 验证要求

1. 推理服务返回的 win rate 与直接加载模型的结果**逐位一致**（同输入）
2. 实测 HTTP 往返延迟，确认对路由总延迟影响可忽略
3. 服务需支持 batch（MMLU 14000 题逐条请求不现实）

## 相关

- 方案文档：`jobfinding/projects/RouteLLM-优化改造方案.md`（4.5 Docker 化、问题5 部署基建）
- 实验记录：`docs/experiments/2026-09-16-bert-router-validation.md`
