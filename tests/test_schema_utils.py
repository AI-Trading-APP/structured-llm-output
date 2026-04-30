import json

from structured_llm_output.schema_utils import pydantic_to_json_schema


def test_screener_filter_schema_basic(schemas):
    schema = pydantic_to_json_schema(schemas["ScreenerFilter"])
    assert schema["type"] == "object"
    assert "properties" in schema
    props = schema["properties"]
    assert "market_cap_min" in props
    assert "sector" in props


def test_trader_rating_schema_includes_enum(schemas):
    schema = pydantic_to_json_schema(schemas["TraderRating"])
    blob = json.dumps(schema)
    assert "Buy" in blob
    assert "Overweight" in blob
    assert "Sell" in blob


def test_research_plan_nested(schemas):
    schema = pydantic_to_json_schema(schemas["ResearchPlan"])
    blob = json.dumps(schema)
    assert "actions" in blob
    assert "thesis" in blob
    assert "risks" in blob
