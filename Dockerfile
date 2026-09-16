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
# 构建（在 43 号机）：
#   docker build -t routellm-eng:dev .
#
# 网络约束（43 号机实测，见 docs/CHANGELOG.md）：
#   - 无法直连 Docker Hub → 基础镜像走 docker.m.daocloud.io
#   - 无法直连 deb.debian.org → Debian 源换阿里云镜像
#   - 容器内 DNS 可能只返回 IPv6 而本机无 IPv6 出口 → 强制 Python 走 IPv4

FROM docker.m.daocloud.io/library/python:3.10-slim

LABEL org.opencontainers.image.title="routellm-eng" \
      org.opencontainers.image.description="RouteLLM gateway（模型推理下沉至 host）"

# ---- Debian 源换国内镜像 + 装最小系统依赖 ---------------------------------
# 只装 curl（健康检查用）。不装 gcc/g++ —— 依赖均为预编译 wheel。
# 初版误装 gcc 导致 apt 阶段卡死 6 分钟以上。
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# ---- Python 源换清华镜像 --------------------------------------------------
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip config set global.timeout 60 \
    && pip config set global.retries 5

# 强制 Python 解析只保留 IPv4（pip 依赖 urllib3，无 Happy Eyeballs 回退；
# 本机无 IPv6 出口时会导致 pip 无限等待）
RUN printf '%s\n' \
    'import socket as _s' \
    '_orig_gai = _s.getaddrinfo' \
    'def _gai_v4(*args, **kwargs):' \
    '    try:' \
    '        res = _orig_gai(*args, **kwargs)' \
    '    except Exception:' \
    '        return _orig_gai(*args, **kwargs)' \
    '    v4 = [r for r in res if r[0] == _s.AF_INET]' \
    '    return v4 or res' \
    '_s.getaddrinfo = _gai_v4' \
    > /usr/local/lib/python3.10/sitecustomize.py

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
