# structured-llm-output

[![test](https://github.com/AI-Trading-APP/structured-llm-output/actions/workflows/test.yml/badge.svg)](https://github.com/AI-Trading-APP/structured-llm-output/actions/workflows/test.yml)
[![distribution](https://img.shields.io/badge/distribution-git%2Bhttps-blue)](#install-consumer-service)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Pydantic-bound provider-native structured output for LLM calls. One small library, three providers (Anthropic + OpenAI + Gemini), one consistent typed-object return shape.

Foundation library (Building Block 1) for the [TradingAgents Adoption Program](https://github.com/AI-Trading-APP/AITradingAPP/issues/101) in `AI-Trading-APP/AITradingAPP`. Used today by `ScreenerService.parse_with_claude` and `ReasoningService` (BB2 + BB4); designed to be reused by future LLM-bound services in the org.

> **Status**: `v0.2.0` shipped 2026-05-04 (added Gemini). See [CHANGELOG.md](CHANGELOG.md). Operations: [RUNBOOK.md](RUNBOOK.md).

## What it does

- Binds a Pydantic v2 schema to an LLM provider's native structured-output mode (Anthropic `tool_use`, OpenAI `json_schema` strict, Gemini `response_schema`)
- Returns a validated, typed instance — no JSON parsing, no manual try/except
- Single corrective retry on parse/validation failure (no retry on provider errors — caller's call)
- One OpenTelemetry span per call with token, cost, schema, and retry-count attributes
- Module-level singleton SDK clients for HTTP keep-alive across consumer concurrency

## Quick start

```python
from structured_llm_output import call_structured, MarkdownRenderable

class Rating(MarkdownRenderable):
    ticker: str
    score: float
    reason: str
    def to_markdown(self) -> str:
        return f"**{self.ticker}**: {self.score:.2f} — {self.reason}"

result: Rating = call_structured(
    model_class=Rating,
    prompt="Rate NVDA on a 0-1 scale",
    provider="anthropic",
    llm_model="claude-haiku-4-5-20251001",
)
print(result.to_markdown())
```

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in the environment. For Gemini, see "Gemini setup" below.

## Install (consumer service)

In your service's `requirements.txt`:

```
structured-llm-output @ git+https://github.com/AI-Trading-APP/structured-llm-output.git@v0.2.0
```

For Gemini support, request the `gemini` extra:

```
structured-llm-output[gemini] @ git+https://github.com/AI-Trading-APP/structured-llm-output.git@v0.2.0
```

This pulls `anthropic`, `openai`, `pydantic` and `opentelemetry-api` transitively (plus `google-genai` if the `[gemini]` extra is selected). Same install pattern as `ai-trading-common` — no Docker path-mounts, no editable installs.

## Gemini setup

The Gemini provider auto-selects backend via env vars (matches the official google-genai SDK convention):

| Backend | Env vars | When to use |
|---|---|---|
| **Vertex AI** | `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT=<project-id>`, `GOOGLE_CLOUD_LOCATION=us-central1`. Auth via `gcloud auth application-default login` or a service account JSON in `GOOGLE_APPLICATION_CREDENTIALS`. | Production. Bills through your GCP project. Enterprise IAM, regional deployment. |
| **AI Studio** | `GOOGLE_API_KEY=<key>` (no Vertex env vars set). | Quick local testing. Bills through Google AI Studio. |

```python
from structured_llm_output import call_structured

result = call_structured(
    model_class=MySchema,
    prompt="...",
    provider="gemini",
    llm_model="gemini-2.5-flash",  # or "gemini-2.5-pro"
)
```

For local dev (e.g., before tagging a new release):

```bash
pip install -e .
```

## Error handling

```python
from structured_llm_output import (
    call_structured,
    StructuredOutputError,            # base
    StructuredOutputValidationError,  # schema/parse failed (retry exhausted)
    StructuredOutputProviderError,    # rate limit, 5xx, timeout — never retried
)

try:
    result = call_structured(...)
except StructuredOutputProviderError as e:
    # Apply YOUR backoff / circuit breaker here. Library deliberately doesn't.
    log.warning("Provider %s returned %s", e.provider, e.status_code)
except StructuredOutputValidationError as e:
    # Both attempts and their validation_errors available on e.attempts
    log.error("LLM couldn't conform to %s: %s", e.schema_name, e.validation_errors)
```

## Run tests

```bash
pip install -e ".[dev]"
pytest -v
```

40 unit tests, 99% coverage on Anthropic+OpenAI core, 94% on the Gemini provider. All provider SDKs are stubbed — no live LLM calls, no API keys required for the default test run.

For optional live-LLM smoke tests:

```bash
pytest -m live_llm   # requires real API keys; excluded from default run
```

## Documentation

| | |
|---|---|
| **Source** | this repo, `src/structured_llm_output/` |
| **Spec / requirements** | [AITradingAPP/specs/structured-llm-output/spec.md](https://github.com/AI-Trading-APP/AITradingAPP/blob/development/specs/structured-llm-output/spec.md) |
| **Design (architecture)** | [AITradingAPP/specs/structured-llm-output/design.md](https://github.com/AI-Trading-APP/AITradingAPP/blob/development/specs/structured-llm-output/design.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |
| **Runbook** | [RUNBOOK.md](RUNBOOK.md) |
| **Issues** | [GitHub Issues](https://github.com/AI-Trading-APP/structured-llm-output/issues) |

## License

[MIT](LICENSE)
