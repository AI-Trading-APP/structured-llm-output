"""Gemini provider tests — mirror the OpenAI provider test pattern.

The Gemini path uses response_schema + JSON mime type (rather than
Anthropic's tool_use), so the response shape under test is `response.text`
(JSON string), not a tool-use block.
"""
from __future__ import annotations

import json

import pytest

from structured_llm_output import (
    StructuredOutputProviderError,
    StructuredOutputValidationError,
    call_structured,
)
from structured_llm_output.providers.gemini import _strip_unsupported_schema_fields


def test_gemini_screener_filter_happy(stub_gemini, schemas, gd):
    """GD-SLO-001 happy path through the Gemini provider."""
    scenario = gd("GD-SLO-001")
    expected = scenario["expected_parsed"]
    stub_gemini["content"] = json.dumps(expected)

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="gemini",
        llm_model="gemini-2.5-flash",
    )

    assert result.market_cap_max == expected["market_cap_max"]
    assert result.sector == expected["sector"]
    assert result.pe_max == expected["pe_max"]


def test_gemini_research_plan_nested(stub_gemini, schemas, gd):
    """GD-SLO-003 — complex nested schema round-trips through Gemini."""
    scenario = gd("GD-SLO-003")
    expected = scenario["expected_parsed"]
    stub_gemini["content"] = json.dumps(expected)

    result = call_structured(
        model_class=schemas["ResearchPlan"],
        prompt=scenario["input"]["prompt"],
        provider="gemini",
        llm_model="gemini-2.5-pro",
    )
    assert result.ticker == expected["ticker"]
    assert result.rating.value == expected["rating"]
    assert result.actions.position_pct == expected["actions"]["position_pct"]


def test_gemini_invalid_json_triggers_retry(stub_gemini, schemas, gd):
    """Malformed JSON on first call → corrective retry (BB1's _MAX_RETRIES=1)."""
    scenario = gd("GD-SLO-001")
    expected = scenario["expected_parsed"]

    # First call returns malformed JSON, second call returns valid
    stub_gemini["contents"] = [
        '{ "market_cap_max": ',  # truncated, JSONDecodeError
        json.dumps(expected),
    ]

    result = call_structured(
        model_class=schemas["ScreenerFilter"],
        prompt=scenario["input"]["prompt"],
        provider="gemini",
        llm_model="gemini-2.5-flash",
    )
    assert result.market_cap_max == expected["market_cap_max"]


def test_gemini_invalid_json_after_retry_raises_validation_error(stub_gemini, schemas, gd):
    """Two malformed responses → StructuredOutputValidationError with both attempts."""
    scenario = gd("GD-SLO-001")

    stub_gemini["contents"] = [
        "not json at all",
        "still not json",
    ]

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt=scenario["input"]["prompt"],
            provider="gemini",
            llm_model="gemini-2.5-flash",
        )
    err = exc_info.value
    assert err.provider == "gemini"
    assert len(err.attempts) == 2


def test_gemini_json_array_root_is_parse_failure(stub_gemini, schemas, gd):
    """JSON array (not object) at root → _ProviderParseFailure → retry → parse fail again."""
    scenario = gd("GD-SLO-001")

    stub_gemini["contents"] = ["[1, 2, 3]", "[4, 5]"]

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt=scenario["input"]["prompt"],
            provider="gemini",
            llm_model="gemini-2.5-flash",
        )
    # Both attempts failed at the parse step (root not object)
    assert all(
        a["validation_errors"][0]["msg"].startswith("json_root_not_object")
        for a in exc_info.value.attempts
    )


def test_gemini_api_error_raises_provider_error(stub_gemini, schemas, gd):
    """SDK exception → StructuredOutputProviderError (not a parse-retry)."""
    from google.genai.errors import APIError

    scenario = gd("GD-SLO-001")
    # Fake APIError; constructor signature: APIError(code, response_json, response)
    stub_gemini["exception"] = APIError(429, {"error": {"message": "rate limited"}}, None)

    with pytest.raises(StructuredOutputProviderError) as exc_info:
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt=scenario["input"]["prompt"],
            provider="gemini",
            llm_model="gemini-2.5-flash",
        )
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.llm_model == "gemini-2.5-flash"


def test_gemini_unrelated_exception_propagates(stub_gemini, schemas, gd):
    """Non-Google exception (programming bug) propagates raw — not wrapped."""
    scenario = gd("GD-SLO-001")
    stub_gemini["exception"] = ValueError("test bug, not a network error")

    with pytest.raises(ValueError, match="test bug"):
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt=scenario["input"]["prompt"],
            provider="gemini",
            llm_model="gemini-2.5-flash",
        )


def test_strip_unsupported_schema_fields_drops_keys():
    """_strip_unsupported_schema_fields removes Pydantic-emitted but
    Gemini-rejected keys: additionalProperties, $schema, default, title."""
    schema = {
        "type": "object",
        "title": "Foo",
        "additionalProperties": False,
        "$schema": "http://json-schema.org/...",
        "properties": {
            "x": {"type": "integer", "default": 0, "title": "X field"},
            "y": {
                "type": "array",
                "items": {"type": "string", "title": "Item"},
            },
        },
    }
    cleaned = _strip_unsupported_schema_fields(schema)
    assert "title" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "$schema" not in cleaned
    assert "default" not in cleaned["properties"]["x"]
    assert "title" not in cleaned["properties"]["x"]
    # Recursion into items
    assert "title" not in cleaned["properties"]["y"]["items"]
    # Other keys preserved
    assert cleaned["type"] == "object"
    assert cleaned["properties"]["x"]["type"] == "integer"


def test_strip_unsupported_schema_fields_handles_lists():
    """Schema fields containing lists of dicts (e.g. anyOf) are recursed."""
    schema = {
        "anyOf": [
            {"type": "string", "title": "Str"},
            {"type": "integer", "title": "Int"},
        ],
    }
    cleaned = _strip_unsupported_schema_fields(schema)
    assert all("title" not in s for s in cleaned["anyOf"])


def test_strip_unsupported_schema_fields_handles_non_dict_input():
    """Defensive: non-dict input returned unchanged."""
    assert _strip_unsupported_schema_fields("not a dict") == "not a dict"  # type: ignore[arg-type]
    assert _strip_unsupported_schema_fields(None) is None  # type: ignore[arg-type]
