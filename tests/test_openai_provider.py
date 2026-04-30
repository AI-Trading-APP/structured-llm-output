from structured_llm_output import call_structured


def test_gd_slo_002_blue_chip_energy(stub_openai, schemas, gd):
    """GD-SLO-002: OpenAI + simple schema, valid response."""
    scenario = gd("GD-SLO-002")
    stub_openai["content"] = scenario["provider_response"]["content"]

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="openai",
        llm_model=scenario["input"]["llm_model"],
    )

    expected = scenario["expected_parsed"]
    assert result.market_cap_min == expected["market_cap_min"]
    assert result.sector == expected["sector"]
    assert result.exchange == expected["exchange"]
    assert result.pe_max is None
