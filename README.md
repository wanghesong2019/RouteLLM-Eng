# 🚀 RouteLLM-Eng

> Production-Grade LLM Routing Gateway based on LMSYS [RouteLLM](https://github.com/lmsys/routellm)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-315%20passed-brightgreen.svg)](#tests)
[![Docker](https://img.shields.io/badge/docker-675MB-blue.svg)](#quick-start)

**Transforming LMSYS RouteLLM from an academic prototype into a production-grade routing gateway.** Adds enterprise-grade caching, circuit breakers, full-chain observability, and async high-concurrency architecture — while preserving routing effectiveness.

## ✨ Key Features

| Feature | What it does |
|---------|-------------|
| ⚡ **Performance** | Identified and refactored `LogisticRegression.fit` bottleneck (→ `newton-cholesky`), cutting routing latency from 394ms to 185ms. Win-rate multi-tier cache hits reduce latency by 4 orders of magnitude. |
| 🛡️ **Resilience** | Three-state circuit breaker + exponential backoff retry + four-level fallback chain (strong → weak → cache → 503). The gateway never cascades. |
| 📊 **Observability** | Non-invasive FastAPI middleware collects per-request metrics → SQLite → built-in ECharts dashboard. Metrics persist across restarts. |
| 🔄 **Async** | Router base class fully async + `httpx.AsyncClient` connection pooling. Slow requests no longer block the event loop. |
| ⚙️ **Hot Reload** | Strong/weak model `base_url`, `api_key`, model names support runtime updates without restart. Immutable config objects with atomic replacement for thread safety. |
| 🔒 **Security** | `/v1/models` endpoint; API key middleware (Bearer + HMAC anti-timing-attack); strict whitelist for ops endpoints. |

## 📐 Architecture

```
              ┌────────────────────────────────┐
              │  Client (OpenAI-compatible)     │
              │  base_url + api_key             │
              └──────────┬─────────────────────┘
                         │ OpenAI API + Bearer key
                         ▼
┌──────────────────────────────────────────────────────────┐
│                RouteLLM Gateway (:6060)                  │
│                                                          │
│  ApiKey Middleware → Metrics Middleware → FastAPI Server │
│  /v1/chat/completions  /metrics  /dashboard             │
│                                                          │
│  ConfigStore (hot reload) ←→ Controller (lock-free read) │
│  SQLite (metrics persistence)                            │
│  MultiTierCache (L1 LRU → L2/L3 injectable)              │
│  Routers: remote_bert (host:6070) / random              │
└──────┬───────────────────────────┬──────────────────────┘
       │                            │ read-only mount
       ▼                            ▼
┌──────────────┐          ┌──────────────────────────┐
│ Strong/Weak  │          │ Dashboard (:8092)        │
│ LLM (OpenAI  │          │ HTML + ECharts           │
│ compatible)  │          │ + Config editor tab      │
└──────────────┘          └──────────────────────────┘
```

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env       # Fill in strong/weak models and credentials
docker compose up -d       # Gateway :6060 + Dashboard :8092
```

The gateway exposes an OpenAI-compatible API — any compatible client works:

```bash
curl http://localhost:6060/v1/chat/completions \
  -H "Authorization: Bearer $ROUTELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"router-bert-0.5",
       "messages":[{"role":"user","content":"..."}]}'
```

The `model` field is a **routing spec** (`router-<name>-<threshold>`), not a downstream model name.

### From source

```bash
pip install -e ".[serve,eval]"
python -m routellm.openai_server --routers random  # random needs no GPU
```

> **Drop-in Replacement**: Since the gateway preserves OpenAI-compatible protocol and upstream interfaces, switching from OpenAI or vanilla RouteLLM requires only changing `base_url`. Zero code changes.

## Why RouteLLM-Eng?

Upstream RouteLLM is an academic prototype — it guarantees algorithm correctness but ignores production concerns:

| Problem | Upstream | RouteLLM-Eng |
|---------|----------|-------------|
| No caching | Every request recalculates win-rate (~350ms) | Multi-tier cache, hit 0.02ms (4 orders of magnitude) |
| No observability | `logging.info` + in-memory dict | FastAPI middleware → SQLite → ECharts dashboard |
| No fault tolerance | Bare `litellm.completion()`, no retry/timeout | Circuit breaker + retry + 4-level fallback chain |
| Sync blocking | Async FastAPI but sync routers | Full async: `httpx.AsyncClient` + `asyncio.to_thread` |
| No deployment | No Dockerfile/CI | Docker (675MB, no torch) + compose dual-container |
| Config frozen | Restart to change models | Runtime hot reload, atomic config swap |
| Broken defaults | Default model provider removed by LiteLLM | All config via env vars, fail-fast validation |

## Evaluation

Reproduces paper metrics (APGR / CPT framework, RouteLLM, ICLR 2025):

| Dataset | APGR | 95% CI | CPT(50%) |
|---------|------|--------|----------|
| MMLU | 0.5328 | [0.5157, 0.5496] | 44.30% |
| GSM8K | 0.5294 | [0.4975, 0.5646] | 45.34% |

Random baseline APGR ≈ 0.5 — **APGR > 0.5 means routing is effective**.

> **Counter-intuitive finding**: APGR cannot be used to select thresholds. PGR increases monotonically with strong-model usage, so maximizing APGR always yields "route everything to strong." Use **CPT** (minimum strong ratio to reach target PGR) instead — fix quality target, then solve for cost.

## Documentation

| Path | Content |
|------|---------|
| `docs/CHANGELOG.md` | Engineering log: every step, measured data, pitfalls |
| `docs/decisions/` | Architecture Decision Records (ADR) |
| `scripts/README.md` | Script index |
| `services/README.md` | Inference service setup |

## Open Source Hygiene

Built-in guard script checks for credentials, internal IPs, and deployment artifacts:

```bash
python scripts/check_open_source_hygiene.py        # Scan workspace
python scripts/check_open_source_hygiene.py --all  # Also scan git history
```

## Tests

```bash
pytest tests/ -q
# 315 passed, 17 skipped
```

## 🗺️ Roadmap

- [ ] Multi-router strategy dynamic switching (BERT, Embedding, etc.)
- [ ] Prometheus / Grafana standard metrics export
- [ ] Streaming response routing optimization
- [ ] MT-Bench evaluation
- [ ] Bilingual dashboard (i18n)

## 🤝 Contributing

Contributions welcome! Before submitting a PR:

1. Run `python scripts/check_open_source_hygiene.py` — ensure no sensitive info
2. Ensure `pytest tests/ -q` passes
3. Reference `docs/decisions/` ADRs for architecture context

## License

MIT License (inherited from upstream), see `LICENSE`.

## Citation

```bibtex
@inproceedings{ong2025routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M Waleed and Stoica, Ion},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
