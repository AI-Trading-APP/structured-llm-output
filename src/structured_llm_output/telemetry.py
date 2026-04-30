from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace

# (provider, llm_model) -> (input_$_per_1M_tokens, output_$_per_1M_tokens)
# Best-effort. Missing entries → cost = 0 + cost_unknown=true.
_PRICING: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-haiku-4-5-20251001"): (1.00, 5.00),
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
    ("anthropic", "claude-opus-4-7"): (15.00, 75.00),
    ("openai", "gpt-4o-2024-11-20"): (2.50, 10.00),
}


def _estimate_cost_usd(
    provider: str, llm_model: str, tokens_in: int, tokens_out: int
) -> tuple[float, bool]:
    pricing = _PRICING.get((provider, llm_model))
    if pricing is None:
        return 0.0, False
    in_rate, out_rate = pricing
    return (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000, True


@contextmanager
def llm_span(provider: str, llm_model: str, schema_name: str) -> Iterator[dict[str, Any]]:
    """Open one OTel span per call_structured invocation.

    Yields a mutable record the caller fills in:
      record['tokens_in'] / record['tokens_out']  — set after provider call
      record['parse_success']                      — set when validation finishes
      record['retry_count']                        — set if retry happens (PR-B)

    Final attributes are written when the context exits, so the caller may
    update fields up to the moment of return or exception.
    """
    # Lazy tracer fetch — picks up any TracerProvider set by the consumer / tests.
    tracer = trace.get_tracer("structured_llm_output", "0.1.0")
    with tracer.start_as_current_span("llm.structured_call") as span:
        record: dict[str, Any] = {
            "tokens_in": 0,
            "tokens_out": 0,
            "parse_success": False,
            "retry_count": 0,
        }
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.model", llm_model)
        span.set_attribute("llm.structured.schema_name", schema_name)
        try:
            yield record
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e)[:500])
            _finalize(span, provider, llm_model, record)
            raise
        else:
            _finalize(span, provider, llm_model, record)


def _finalize(span: Any, provider: str, llm_model: str, record: dict[str, Any]) -> None:
    cost, known = _estimate_cost_usd(
        provider, llm_model, record["tokens_in"], record["tokens_out"]
    )
    span.set_attribute("llm.tokens_in", record["tokens_in"])
    span.set_attribute("llm.tokens_out", record["tokens_out"])
    span.set_attribute("llm.cost_usd", cost)
    if not known:
        span.set_attribute("llm.cost_unknown", True)
    span.set_attribute("llm.structured.parse_success", record["parse_success"])
    span.set_attribute("llm.structured.retry_count", record["retry_count"])
