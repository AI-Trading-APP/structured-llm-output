from structured_llm_output import call_structured


def test_gd_slo_001_screener_filter(stub_anthropic, schemas, gd):
    """GD-SLO-001: Anthropic + simple schema, valid response."""
    scenario = gd("GD-SLO-001")
    stub_anthropic["response"] = scenario["provider_response"]

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="anthropic",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.market_cap_max == expected["market_cap_max"]
    assert result.sector == expected["sector"]
    assert result.pe_max == expected["pe_max"]
    assert result.market_cap_min is None


def test_gd_slo_003_research_plan_nested(stub_anthropic, schemas, gd):
    """GD-SLO-003: Anthropic + complex nested schema, valid response."""
    scenario = gd("GD-SLO-003")
    stub_anthropic["response"] = scenario["provider_response"]

    result = call_structured(
        model_class=schemas["ResearchPlan"],
        prompt=scenario["input"]["prompt"],
        provider="anthropic",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.ticker == expected["ticker"]
    assert result.rating.value == expected["rating"]
    assert result.thesis == expected["thesis"]
    assert result.risks == expected["risks"]
    assert result.actions.entry_price == expected["actions"]["entry_price"]
    assert result.actions.stop_loss == expected["actions"]["stop_loss"]
    assert result.actions.position_pct == expected["actions"]["position_pct"]


def test_gd_slo_006_extra_field_ignored(stub_anthropic, schemas, gd):
    """GD-SLO-006: Extra unknown field silently dropped, validation succeeds."""
    scenario = gd("GD-SLO-006")
    stub_anthropic["response"] = scenario["provider_response"]

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="anthropic",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.sector == expected["sector"]
    assert result.pe_max == expected["pe_max"]
    assert not hasattr(result, "rd_intensity_min")
