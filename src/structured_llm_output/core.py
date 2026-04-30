from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import ValidationError

from ._internal import _ProviderParseFailure
from .exceptions import StructuredOutputValidationError
from .providers import anthropic as anthropic_provider
from .providers import openai as openai_provider
from .renderable import MarkdownRenderable
from .telemetry import llm_span

T = TypeVar("T", bound=MarkdownRenderable)

_RESERVED_PROVIDER_KEYS = frozenset({
    "tools", "tool_choice", "response_format",
    "messages", "model", "max_tokens", "system",
})


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

    PR-A scope: no retry — validation failures raise immediately.
    Retry semantics ship in PR-B (T8) per tasks.md.

    Raises:
        ValueError                          — unknown provider, or reserved provider_kwargs key
        TypeError                           — model_class doesn't extend MarkdownRenderable, or
                                              missing to_markdown override
        StructuredOutputValidationError     — schema validation failed, OR provider returned
                                              unparseable output (no tool_use block, non-JSON)
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

    with llm_span(provider, llm_model, model_class.__name__) as record:
        try:
            if provider == "anthropic":
                parsed, tokens_in, tokens_out = anthropic_provider.call_anthropic(
                    model_class=model_class,
                    prompt=prompt,
                    system=system,
                    llm_model=llm_model,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    provider_kwargs=pkwargs,
                )
            else:
                parsed, tokens_in, tokens_out = openai_provider.call_openai(
                    model_class=model_class,
                    prompt=prompt,
                    system=system,
                    llm_model=llm_model,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    provider_kwargs=pkwargs,
                )
        except _ProviderParseFailure as e:
            record["tokens_in"] = e.tokens_in
            record["tokens_out"] = e.tokens_out
            raise StructuredOutputValidationError(
                f"{model_class.__name__} parse failure: {e.reason}",
                provider=provider,
                llm_model=llm_model,
                schema_name=model_class.__name__,
                raw_response=e.raw_response,
                validation_errors=[{"loc": [], "msg": e.reason, "type": "provider_parse"}],
                attempts=[{"raw_response": e.raw_response, "validation_errors": []}],
            ) from e

        record["tokens_in"] = tokens_in
        record["tokens_out"] = tokens_out
        try:
            instance = model_class.model_validate(parsed)
        except ValidationError as e:
            errors = e.errors()
            raise StructuredOutputValidationError(
                f"{model_class.__name__} validation failed: {len(errors)} error(s)",
                provider=provider,
                llm_model=llm_model,
                schema_name=model_class.__name__,
                raw_response=parsed,
                validation_errors=errors,
                attempts=[{"raw_response": parsed, "validation_errors": errors}],
            ) from e
        record["parse_success"] = True
        return instance
