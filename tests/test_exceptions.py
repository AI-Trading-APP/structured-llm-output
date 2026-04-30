from structured_llm_output import (
    StructuredOutputError,
    StructuredOutputProviderError,
    StructuredOutputValidationError,
)


def test_validation_error_attrs():
    e = StructuredOutputValidationError(
        "boom",
        provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
        schema_name="X",
        raw_response={"foo": "bar"},
        validation_errors=[{"loc": ("x",), "msg": "missing", "type": "missing"}],
    )
    assert e.provider == "anthropic"
    assert e.schema_name == "X"
    assert e.raw_response == {"foo": "bar"}
    assert e.validation_errors[0]["msg"] == "missing"
    assert e.attempts == []
    assert isinstance(e, StructuredOutputError)


def test_validation_error_carries_attempts():
    e = StructuredOutputValidationError(
        "boom",
        provider="openai",
        llm_model="gpt-4o-2024-11-20",
        schema_name="Y",
        raw_response={"a": 1},
        validation_errors=[],
        attempts=[
            {"raw_response": {"a": 1}, "validation_errors": []},
            {"raw_response": {"a": 2}, "validation_errors": []},
        ],
    )
    assert len(e.attempts) == 2


def test_provider_error_attrs():
    underlying = RuntimeError("rate limit")
    e = StructuredOutputProviderError(
        "boom",
        provider="anthropic",
        llm_model="claude-haiku",
        schema_name="X",
        underlying_exception=underlying,
        status_code=429,
    )
    assert e.status_code == 429
    assert e.underlying_exception is underlying
    assert isinstance(e, StructuredOutputError)
