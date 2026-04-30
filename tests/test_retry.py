"""Single corrective retry semantics — REQ-SLO-004."""
import pytest

from structured_llm_output import StructuredOutputValidationError, call_structured


def test_gd_slo_004_malformed_json_recovers(stub_openai, schemas, gd):
    """OpenAI returns malformed JSON → retry returns valid JSON → success, retry_count=1."""
    scenario = gd("GD-SLO-004")
    stub_openai["contents"] = [
        scenario["provider_responses_in_order"][0]["content"],
        scenario["provider_responses_in_order"][1]["content"],
    ]

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="openai",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.market_cap_min == expected["market_cap_min"]
    assert result.sector == expected["sector"]


def test_gd_slo_005_missing_field_exhausts_retry(stub_anthropic, schemas, gd):
    """Anthropic returns valid tool_use missing required field → retry same → raise.

    Verifies attempts list has 2 entries.
    """
    scenario = gd("GD-SLO-005")
    stub_anthropic["responses"] = scenario["provider_responses_in_order"]

    with pytest.raises(StructuredOutputValidationError) as exc:
        call_structured(
            model_class=schemas["TraderRating"],
            prompt=scenario["input"]["prompt"],
            provider="anthropic",
            llm_model=scenario["input"]["llm_model"],
        )

    assert len(exc.value.attempts) == 2
    # Both attempts had the same missing-field validation error
    err_msgs = [
        v["msg"] for attempt in exc.value.attempts for v in attempt["validation_errors"]
    ]
    assert any("missing" in msg.lower() or "required" in msg.lower() for msg in err_msgs)


def test_gd_slo_007_enum_out_of_set_exhausted(stub_openai, schemas, gd):
    """OpenAI returns invalid enum twice → raise after retry."""
    scenario = gd("GD-SLO-007")
    stub_openai["contents"] = [
        scenario["provider_responses_in_order"][0]["content"],
        scenario["provider_responses_in_order"][1]["content"],
    ]

    with pytest.raises(StructuredOutputValidationError) as exc:
        call_structured(
            model_class=schemas["TraderRating"],
            prompt=scenario["input"]["prompt"],
            provider="openai",
            llm_model=scenario["input"]["llm_model"],
        )

    assert len(exc.value.attempts) == 2


def test_gd_slo_008_confidence_out_of_range_recovers(stub_anthropic, schemas, gd):
    """Anthropic returns confidence=1.4 → retry returns 0.82 → success, retry_count=1."""
    scenario = gd("GD-SLO-008")
    stub_anthropic["responses"] = scenario["provider_responses_in_order"]

    result = call_structured(
        model_class=schemas["TraderRating"],
        prompt=scenario["input"]["prompt"],
        provider="anthropic",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.ticker == expected["ticker"]
    assert result.rating.value == expected["rating"]
    assert result.confidence == expected["confidence"]


def test_retry_count_recorded_on_recovery(otel_exporter, stub_anthropic, schemas, gd):
    """When retry recovers, OTel span records retry_count=1 and parse_success=True."""
    scenario = gd("GD-SLO-008")
    stub_anthropic["responses"] = scenario["provider_responses_in_order"]

    call_structured(
        model_class=schemas["TraderRating"],
        prompt=scenario["input"]["prompt"],
        provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
    )

    span = otel_exporter.get_finished_spans()[0]
    attrs = dict(span.attributes)
    assert attrs["llm.structured.parse_success"] is True
    assert attrs["llm.structured.retry_count"] == 1
    # Tokens accumulated across both calls
    assert attrs["llm.tokens_in"] == 200  # 100 per stubbed call
    assert attrs["llm.tokens_out"] == 100
