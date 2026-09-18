# Contributing to RouteLLM-Eng

Thank you for your interest in contributing! This document covers the basics.

## Development Setup

```bash
git clone <repo-url>
cd RouteLLM-Eng
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,eval]"
pip install pytest
```

## Before Submitting a PR

1. **Open source hygiene** — no credentials, internal IPs, or deployment artifacts:
   ```bash
   python scripts/check_open_source_hygiene.py
   ```

2. **All tests pass**:
   ```bash
   pytest tests/ -q
   ```

3. **No new heavy dependencies** — the gateway image stays lightweight (no torch in container). If your change requires a new dependency, document why in the PR.

## Development Principles

- **TDD**: Write tests first (RED), then implement (GREEN). Don't skip this.
- **Minimal changes**: Modify only what's needed. Don't refactor unrelated code.
- **Evidence-based**: Back claims with measured data, not assumptions.

## Architecture Context

Read `docs/decisions/ADR-001-inference-service-on-host.md` to understand why model inference is separated from the gateway container.

## Commit Style

Use conventional commits:
- `feat(scope): description` — new feature
- `fix(scope): description` — bug fix
- `docs: description` — documentation only
- `chore: description` — tooling, config
