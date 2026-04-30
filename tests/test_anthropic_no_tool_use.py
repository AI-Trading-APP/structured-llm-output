"""GD-SLO-013 — Anthropic returns text instead of tool_use, recovers via retry.

Adopted per CPO follow-up M1 + Phase 2 design risk R-BB1-003.
Anthropic occasionally returns a text block under load even when
tool_choice forces a specific tool — library treats this as a
parse failure recoverable via the standard single corrective retry.
"""
from structured_llm_output import call_structured


def test_gd_slo_013_text_block_then_tool_use_recovers(
    stub_anthropic, schemas, gd
):
    scenario = gd("GD-SLO-013")
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
    assert result.rationale == expected["rationale"]


def test_no_tool_use_recovery_records_retry_count(
    otel_exporter, stub_anthropic, schemas, gd
):
    scenario = gd("GD-SLO-013")
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
