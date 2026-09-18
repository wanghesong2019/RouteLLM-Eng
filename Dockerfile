# RouteLLM 网关容器
#
# 设计依据：docs/decisions/ADR-001-inference-service-on-host.md
#
# 核心设计：模型推理已下沉到 host 侧推理服务，本容器只承载**路由逻辑 + 网关**。
# 因此容器内**不安装 torch / transformers**（554MB+），只装网关必需的依赖。
# 使用 remote_bert 或 random 路由器即可完整工作。
#
# 若确实需要在容器内跑进程内推理（bert / causal_llm / sw_ranking / mf），
# 用 docker-compose.full.yml 或构建 full 阶段（见文件末尾注释）。
#
# 构建：
#   docker build -t routellm-eng:dev .

FROM python:3.10-slim

LABEL org.opencontainers.image.title="routellm-eng" \
      org.opencontainers.image.description="RouteLLM gateway（模型推理下沉至 host）"

# ---- 系统依赖 -----------------------------------------------------------
# 只装 curl（健康检查用）。不装 gcc/g++ —— 依赖均为预编译 wheel。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖（网关切片，不含 torch）------------------------------------
WORKDIR /app

COPY requirements-gateway.txt ./
RUN pip install --no-cache-dir -r requirements-gateway.txt

# ---- 应用代码（以可编辑方式装入，便于挂载调试）-----------------------------
COPY routellm/ ./routellm/
COPY pyproject.toml README.md config.example.yaml ./
RUN pip install --no-cache-dir --no-deps -e .

# ---- 运行配置 -------------------------------------------------------------
# 下游 LLM provider（OpenAI 兼容）—— 全部通过环境变量注入
ENV ROUTELLM_STRONG_MODEL=""
ENV ROUTELLM_WEAK_MODEL=""
ENV ROUTELLM_API_BASE=""
# 注：下游 api_key（ROUTELLM_API_KEY）由 compose 在运行时注入，镜像内不设默认值
ENV ROUTELLM_INFERENCE_URL="http://host.docker.internal:6070"
ENV ROUTELLM_ROUTERS="random"
ENV ROUTELLM_PORT="6060"
ENV PYTHONUNBUFFERED="1"

# 非 root 运行
RUN useradd -m -u 1000 routellm && chown -R routellm:routellm /app
USER routellm

EXPOSE 6060

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:6060/health || exit 1

CMD ["python", "-m", "routellm.openai_server", "--host", "0.0.0.0", "--port", "6060"]
