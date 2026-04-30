# Runbook — structured-llm-output

Operational guide for consumer services using this library.

## Health check (production / test)

The library has no service of its own — health is the consumer service's concern. To verify the lib is working from a consumer:

```python
from structured_llm_output import call_structured, MarkdownRenderable

class Pong(MarkdownRenderable):
    pong: str
    def to_markdown(self): return self.pong

result = call_structured(
    model_class=Pong,
    prompt='Reply with {"pong": "ok"}',
    provider="anthropic",
    llm_model="claude-haiku-4-5-20251001",
    max_tokens=64,
)
assert result.pong == "ok"
```

If this fails, troubleshooting tree below.

## Provider outage

**Symptom**: `StructuredOutputProviderError` with `status_code` 429 (rate limit), 500–504 (server),
or no status (timeout). The library does **not** retry these — that's the consumer's responsibility.

**Action**:
1. Check provider status pages — [Anthropic status](https://status.anthropic.com/), [OpenAI status](https://status.openai.com/).
2. If consumer has a fallback (regex parser, cached response, etc.), it should already kick in via the caught exception.
3. If no fallback, consider temporarily switching to the other provider via runtime config.
4. For sustained provider issues, engage on-call to enable degraded-mode UI.

## Rate-limit storm

**Symptom**: persistent `status_code=429` from one provider.

**Action**:
1. Check the consumer's call rate dashboard (`llm.structured_call` span count over time).
2. If burst-driven, add a circuit breaker at the consumer layer (the lib deliberately does not own breaker logic — see [design.md §6.1](https://github.com/AI-Trading-APP/AITradingAPP/blob/development/specs/structured-llm-output/design.md#61-resilience-patterns--explicitly-the-consumers-responsibility)).
3. If sustained, request a rate-limit increase from the provider OR shift traffic by `llm_model` (e.g., move some workload from Sonnet to Haiku).

## Schema-eval regression

**Symptom**: `StructuredOutputValidationError` rate spikes after a model update or prompt change.

**Action**:
1. Check the per-`schema_name` failure rate in OTel — narrow to the affected schema.
2. Inspect a sample of `attempts[*].raw_response` from logs — what shape did the LLM return?
3. If the LLM consistently misses a field, the prompt likely needs more explicit guidance.
4. If the schema added a new required field, downgrade to the previous lib version OR roll the prompt forward.
5. **Never** silently relax the schema (e.g., make a required field optional) without replacing the validation
   with another mechanism — it converts hard failures into silent data corruption.

## OTel exporter offline

**Symptom**: cost dashboard / trace queries return no data, but service is otherwise healthy.

**Action**:
1. The lib only depends on `opentelemetry-api`. If the consumer hasn't initialized the OTel **SDK**,
   spans become no-ops. Confirm the consumer service has SDK init code.
2. Check `OTEL_EXPORTER_OTLP_ENDPOINT` env var on the consumer.
3. Library calls continue to work even if OTel export fails — there is no hard dependency.

## Cost-per-call drift

**Symptom**: `llm.cost_usd` attribute appears to be incorrect compared to provider invoice.

**Action**:
1. Check `llm.cost_unknown=true` rate. If high, the model isn't in the lib's pricing table — file an
   issue or PR to update [`telemetry.py`](src/structured_llm_output/telemetry.py) `_PRICING` dict.
2. Even when known, the pricing table is **best-effort** and can lag provider price changes by days.
   Reconcile monthly against the actual invoice.
3. For audit-grade cost tracking, query the provider's billing API directly — the library is optimized
   for real-time signals, not invoice reconciliation.

## Disabling structured output entirely (kill switch)

There is no library-level kill switch. Each consumer should:
1. Wrap `call_structured` in a feature flag check.
2. On flag-off, fall back to the consumer's pre-existing parsing path (regex, hand-rolled, etc.).

For BB1 example: ScreenerService has `parse_screener_prompt_ai` which already falls back to
`screener_nlp.py` regex parsing on any failure — disabling the lib effectively just means
`parse_with_claude` always returns None.

## Versioning & upgrade

This package uses [SemVer](https://semver.org). Bump rules:

| Change | Version bump |
|---|---|
| Adding a new schema, new optional kwarg, new exception field | minor (0.1.x → 0.2.0) |
| Breaking signature change, removing a public name, renaming an exception | major (0.x → 1.0) |
| Bug fix with no contract change | patch (0.1.0 → 0.1.1) |

Consumers pin to a specific tag in their `requirements.txt`:

```
structured-llm-output @ git+https://github.com/AI-Trading-APP/structured-llm-output.git@v0.1.0
```

To upgrade: change the tag, rebuild the consumer's Docker image / re-pip-install.

## Where to file issues

- Bugs / behavior questions → [structured-llm-output issues](https://github.com/AI-Trading-APP/structured-llm-output/issues)
- Architecture / cross-program concerns → [AITradingAPP issues](https://github.com/AI-Trading-APP/AITradingAPP/issues) tagged `ts-llm-program`
