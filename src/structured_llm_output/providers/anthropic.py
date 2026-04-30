from __future__ import annotations

import threading
from typing import Any

import anthropic

from .._internal import _ProviderParseFailure
from ..exceptions import StructuredOutputProviderError
from ..renderable import MarkdownRenderable
from ..schema_utils import pydantic_to_json_schema

_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def _get_client(timeout: float) -> anthropic.Anthropic:
    """Module-level singleton for HTTP keep-alive (CTO MAJ-1).

    Thread-safe lazy init. The Anthropic SDK's underlying httpx client is
    documented as thread-safe with internal connection pooling, so a single
    instance suffices for any consumer concurrency model.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = anthropic.Anthropic(timeout=timeout)
    return _client


def reset_client() -> None:
    """Drop the singleton — for test isolation."""
    global _client
    with _client_lock:
        _client = None


def call_anthropic(
    *,
    model_class: type[MarkdownRenderable],
    prompt: str,
    system: str | None,
    llm_model: str,
    max_tokens: int,
    timeout_seconds: float,
    provider_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """Anthropic Messages API with forced tool_use binding.

    Returns: (parsed_input_dict, tokens_in, tokens_out)
    Raises: _ProviderParseFailure if no tool_use block returned (recoverable in PR-B).
            StructuredOutputProviderError on SDK errors (rate limit, 5xx, timeout).
    """
    schema = pydantic_to_json_schema(model_class)
    tool: dict[str, Any] = {
        "name": model_class.__name__,
        "description": (model_class.__doc__ or "Return a structured response.").strip()[:1000],
        "input_schema": schema,
    }
    client = _get_client(timeout_seconds)
    try:
        response = client.messages.create(
            model=llm_model,
            system=system or "Use the provided tool to respond.",
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": model_class.__name__},
            max_tokens=max_tokens,
            **provider_kwargs,
        )
    except anthropic.APIError as e:
        raise StructuredOutputProviderError(
            f"Anthropic API error: {e}",
            provider="anthropic",
            llm_model=llm_model,
            schema_name=model_class.__name__,
            underlying_exception=e,
            status_code=getattr(e, "status_code", None),
        ) from e

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == model_class.__name__:
            return dict(block.input), tokens_in, tokens_out

    raw = response.model_dump() if hasattr(response, "model_dump") else {"content": str(response.content)}
    raise _ProviderParseFailure(
        raw_response=raw,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        reason="no_tool_use_block",
    )
