# Changelog

All notable changes to `structured-llm-output` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-06-17

### Added
- **Multimodal attachments.** `call_structured(..., attachments=[MediaInput(...)])` sends images / PDFs alongside the text prompt for vision-based extraction, while keeping the Pydantic-schema binding and corrective-retry loop unchanged.
- New public `MediaInput(mime_type, data: bytes)` value type (`structured_llm_output.media`). Accepts `image/png`, `image/jpeg`, `image/webp`, `image/gif`, and `application/pdf`; validates on construction.
- Per-provider wire shapes: Anthropic image/`document` blocks (PDF supported), Gemini inline `Part.from_bytes` (PDF supported), OpenAI `image_url` parts (images only — PDF raises `ValueError`).
- Primary consumer: the EPI portfolio-import vision extractor (broker screenshot/PDF → holdings).

### Changed
- All three provider functions accept an optional `attachments` kwarg. The no-attachment path sends the bare prompt string exactly as before (byte-identical wire shape), so existing callers are unaffected.

## [0.2.0] — 2026-05-04

### Added
- **Gemini provider** via `google-genai` SDK. Backend auto-selected by env vars (matches the official google-genai convention):
  - `GOOGLE_GENAI_USE_VERTEXAI=true` + `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` → **Vertex AI** (Application Default Credentials)
  - else `GOOGLE_API_KEY` → **Google AI Studio** direct API
- `provider="gemini"` accepted by `call_structured`. Uses `response_mime_type=application/json` + `response_schema` (Gemini's equivalent of Anthropic's `tool_use` forced binding).
- Optional install extra: `pip install 'structured-llm-output[gemini]'` adds `google-genai>=1.0,<2.0`.
- `_strip_unsupported_schema_fields` helper: drops Pydantic-emitted JSON Schema keys (`additionalProperties`, `$schema`, `default`, `title`) that Gemini's schema validator rejects.

### Changed
- `provider` Literal type widened from `Literal["anthropic", "openai"]` to `Literal["anthropic", "openai", "gemini"]`.
- `tests/conftest.py`: `_resolve_golden_path()` now searches multiple candidate paths (legacy sibling layout, AITradingAPP-monorepo layout, in-repo) so the test suite runs from any clone position. Override via `SLO_GOLDEN_DATA` env var.

### Tests
- 10 new tests covering happy path, retry-on-malformed-JSON, parse-failure-then-retry-then-validation-error, JSON-array-root rejection, APIError → StructuredOutputProviderError mapping, non-Google exception passthrough, and `_strip_unsupported_schema_fields` recursion. **94% line coverage on the new provider.** All 40 tests pass.

### Carry-forwards (non-blocking)
- **MIN-G1** Live Gemini smoke tests gated behind `@pytest.mark.live_llm` not yet added — current 10 tests stub `google.genai.models.Models.generate_content`.
- **MIN-G2** `timeout_seconds` argument is currently advisory for the Gemini path — google-genai's HTTP timeout is set per-call via `http_options`, not on the client. Consumers needing strict timeouts can set it via `provider_kwargs={"http_options": {"timeout": 30000}}` (milliseconds).

## [0.1.0] — 2026-04-30

Initial release. Building Block 1 (BB1) of the
[TradingAgents Adoption Program](https://github.com/AI-Trading-APP/AITradingAPP/issues/101)
in `AI-Trading-APP/AITradingAPP`.

### Added

- `call_structured(model_class, prompt, *, provider, llm_model, ...)` — single public entry point
  that binds a Pydantic v2 schema to an LLM provider's native structured-output mode and returns a
  validated, typed instance.
- Provider support:
  - **Anthropic** Claude family via Messages API tool_use binding (forced tool_choice).
  - **OpenAI** GPT-4o family via Chat Completions `response_format={"type":"json_schema","strict":true}`.
- `MarkdownRenderable` — base Pydantic class all schemas must extend; requires `to_markdown(self) -> str`.
- Typed exception hierarchy:
  - `StructuredOutputError` (base)
  - `StructuredOutputValidationError` — schema validation failed; carries `attempts` list, raw responses, validation_errors
  - `StructuredOutputProviderError` — non-recoverable provider error (rate limit, 5xx, timeout); never retried
- **Single corrective retry** on parse/validation failure with corrective prompt rewrite (errors + schema injected).
- **OpenTelemetry** instrumentation: one span (`llm.structured_call`) per call with `llm.provider`, `llm.model`,
  `llm.tokens_in`, `llm.tokens_out`, `llm.cost_usd`, `llm.structured.schema_name`,
  `llm.structured.parse_success`, `llm.structured.retry_count` attributes.
- Module-level singleton SDK clients per provider for HTTP keep-alive (CTO MAJ-1 from Phase 2 review).
- `provider_kwargs` reserved-key collision detection.
- Explicit `to_markdown` override check at call time.

### Tests
- 30 unit tests, 99% line coverage. Provider SDKs stubbed — no live LLM calls in CI.
- Live-LLM smoke tests gated behind `@pytest.mark.live_llm` for manual / nightly verification.

### Distribution
- Installed via `pip install structured-llm-output @ git+https://github.com/AI-Trading-APP/structured-llm-output.git@v0.1.0`.
- Pattern matches `ai-trading-common` (existing org convention).

### Known carry-forwards (non-blocking)
- **MIN-3** Tighten SDK pins after explicit testing of newer minors. Currently wide
  (`anthropic >=0.40,<1.0`, `openai >=1.50,<2.0`) — tested against 0.86 / 1.97.
- **MIN-4** Document PII redaction caveat for opt-in prompt logging in expanded README / RUNBOOK.
- **MIN-5** Optional `tool_name_override` for class-name leak avoidance — defer until a consumer needs it.

### History
- Source code originally lived at `AI-Trading-APP/AITradingAPP/shared-libs/structured-llm-output/` and was
  developed in PR-A (#104) and PR-B (#105) in that repo. On 2026-04-30, after the user-approved
  distribution-pattern pivot, the directory was extracted via `git subtree split` and pushed here as
  the `main` branch. The two original commits remain in the git log.

[0.2.0]: https://github.com/AI-Trading-APP/structured-llm-output/releases/tag/v0.2.0
[0.1.0]: https://github.com/AI-Trading-APP/structured-llm-output/releases/tag/v0.1.0
