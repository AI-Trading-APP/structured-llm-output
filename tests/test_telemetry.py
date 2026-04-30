from structured_llm_output import call_structured


def test_span_attributes_on_success(otel_exporter, stub_anthropic, schemas, gd):
    scenario = gd("GD-SLO-001")
    stub_anthropic["response"] = scenario["provider_response"]

    call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt="x",
        provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
    )

    spans = otel_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "llm.structured_call"
    attrs = dict(span.attributes)
    assert attrs["llm.provider"] == "anthropic"
    assert attrs["llm.model"] == "claude-haiku-4-5-20251001"
    assert attrs["llm.structured.schema_name"] == "ScreenerFilter"
    assert attrs["llm.structured.parse_success"] is True
    assert attrs["llm.structured.retry_count"] == 0
    assert attrs["llm.tokens_in"] == 100
    assert attrs["llm.tokens_out"] == 50
    assert attrs["llm.cost_usd"] > 0
    assert "llm.cost_unknown" not in attrs


def test_span_marks_unknown_cost_for_unpriced_model(otel_exporter, stub_anthropic, schemas, gd):
    scenario = gd("GD-SLO-001")
    stub_anthropic["response"] = scenario["provider_response"]

    call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt="x",
        provider="anthropic",
        llm_model="claude-future-model-not-yet-priced",
    )

    span = otel_exporter.get_finished_spans()[0]
    attrs = dict(span.attributes)
    assert attrs["llm.cost_unknown"] is True
    assert attrs["llm.cost_usd"] == 0.0


def test_span_records_error_on_validation_failure(
    otel_exporter, stub_anthropic, schemas, gd
):
    """Validation error → span has error.type and parse_success=False."""
    # Force a validation failure: return a response missing required `rating`.
    stub_anthropic["response"] = {
        "type": "tool_use",
        "name": "TraderRating",
        "input": {"ticker": "AAPL", "confidence": 0.7, "rationale": "stable"},
    }

    import pytest
    from structured_llm_output import StructuredOutputValidationError

    with pytest.raises(StructuredOutputValidationError):
        call_structured(
            model_class=schemas["TraderRating"],
            prompt="rate AAPL",
            provider="anthropic",
            llm_model="claude-haiku-4-5-20251001",
        )

    span = otel_exporter.get_finished_spans()[0]
    attrs = dict(span.attributes)
    assert attrs["llm.structured.parse_success"] is False
    assert attrs["error.type"] == "StructuredOutputValidationError"
    assert "validation failed" in attrs["error.message"]
