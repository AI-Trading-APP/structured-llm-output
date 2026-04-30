from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import ValidationError

from ._internal import _ProviderParseFailure
from .exceptions import StructuredOutputValidationError
from .providers import anthropic as anthropic_provider
from .providers import openai as openai_provider
from .renderable import MarkdownRenderable
from .schema_utils import pydantic_to_json_schema
from .telemetry import llm_span

T = TypeVar("T", bound=MarkdownRenderable)

_RESERVED_PROVIDER_KEYS = frozenset({
    "tools", "tool_choice", "response_format",
    "messages", "model", "max_tokens", "system",
})

_MAX_RETRIES = 1


def _build_retry_prompt(
    original_prompt: str,
    last_attempt_errors: list[dict[str, Any]],
    schema_json: dict[str, Any],
) -> str:
    return (
        f"{original_prompt}\n\n"
        f"---\n"
        f"Your previous response did not match the required schema. Errors:\n"
        f"{json.dumps(last_attempt_errors, indent=2, default=str)}\n\n"
        f"Required schema:\n"
        f"{json.dumps(schema_json, indent=2)}\n\n"
        f"Please respond again, conforming exactly to the schema."
    )


def call_structured(
    model_class: type[T],
    prompt: str,
    *,
    provider: Literal["anthropic", "openai"],
    llm_model: str,
    system: str | None = None,
    max_tokens: int = 4096,
    provider_kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> T:
    """Call an LLM and return a validated instance of `model_class`.

    On parse/validation failure, retries exactly once with a corrective prompt
    (REQ-SLO-004). Provider errors (rate limit, 5xx, timeout) are NOT retried —
    they bubble up as StructuredOutputProviderError so callers can apply their
    own backoff or breaker logic (see design §6.1).

    Raises:
        ValueError                          — unknown provider, or reserved provider_kwargs key
        TypeError                           — model_class doesn't extend MarkdownRenderable, or
                                              missing to_markdown override
        StructuredOutputValidationError     — validation/parse failure after the retry attempt;
                                              `attempts` list contains both attempts' raw responses
        StructuredOutputProviderError       — provider error (rate limit, 5xx, timeout)
    """
    if provider not in ("anthropic", "openai"):
        raise ValueError(
            f"provider must be one of: anthropic, openai (got {provider!r})"
        )
    if not (isinstance(model_class, type) and issubclass(model_class, MarkdownRenderable)):
        raise TypeError(
            f"model_class must be a subclass of MarkdownRenderable; got {model_class!r}"
        )
    if model_class.to_markdown is MarkdownRenderable.to_markdown:
        raise TypeError(
            f"{model_class.__name__} must override to_markdown(); "
            "the base MarkdownRenderable.to_markdown raises NotImplementedError."
        )

    pkwargs = dict(provider_kwargs or {})
    bad = pkwargs.keys() & _RESERVED_PROVIDER_KEYS
    if bad:
        raise ValueError(
            f"provider_kwargs contains reserved keys managed by the library: {sorted(bad)}"
        )

    schema_json = pydantic_to_json_schema(model_class)
    provider_fn = (
        anthropic_provider.call_anthropic
        if provider == "anthropic"
        else openai_provider.call_openai
    )

    with llm_span(provider, llm_model, model_class.__name__) as record:
        attempts_log: list[dict[str, Any]] = []
        current_prompt = prompt

        for attempt_idx in range(_MAX_RETRIES + 1):
            try:
                parsed, tokens_in, tokens_out = provider_fn(
                    model_class=model_class,
                    prompt=current_prompt,
                    system=system,
                    llm_model=llm_model,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    provider_kwargs=pkwargs,
                )
            except _ProviderParseFailure as e:
                record["tokens_in"] += e.tokens_in
                record["tokens_out"] += e.tokens_out
                attempt_errors = [{"loc": [], "msg": e.reason, "type": "provider_parse"}]
                attempts_log.append({
                    "raw_response": e.raw_response,
                    "validation_errors": attempt_errors,
                })
                if attempt_idx < _MAX_RETRIES:
                    current_prompt = _build_retry_prompt(prompt, attempt_errors, schema_json)
                    record["retry_count"] = attempt_idx + 1
                    continue
                raise StructuredOutputValidationError(
                    f"{model_class.__name__} parse failure after {_MAX_RETRIES} retry: {e.reason}",
                    provider=provider,
                    llm_model=llm_model,
                    schema_name=model_class.__name__,
                    raw_response=e.raw_response,
                    validation_errors=attempt_errors,
                    attempts=attempts_log,
                ) from e

            record["tokens_in"] += tokens_in
            record["tokens_out"] += tokens_out
            try:
                instance = model_class.model_validate(parsed)
            except ValidationError as e:
                errors = e.errors()
                attempts_log.append({
                    "raw_response": parsed,
                    "validation_errors": errors,
                })
                if attempt_idx < _MAX_RETRIES:
                    current_prompt = _build_retry_prompt(prompt, errors, schema_json)
                    record["retry_count"] = attempt_idx + 1
                    continue
                raise StructuredOutputValidationError(
                    f"{model_class.__name__} validation failed after {_MAX_RETRIES} retry: {len(errors)} error(s)",
                    provider=provider,
                    llm_model=llm_model,
                    schema_name=model_class.__name__,
                    raw_response=parsed,
                    validation_errors=errors,
                    attempts=attempts_log,
                ) from e

            record["parse_success"] = True
            return instance

        raise RuntimeError("call_structured exited retry loop without success or raise")
