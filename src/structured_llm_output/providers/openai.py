from __future__ import annotations

import json
import threading
from typing import Any

import openai

from .._internal import _ProviderParseFailure
from ..exceptions import StructuredOutputProviderError
from ..renderable import MarkdownRenderable
from ..schema_utils import pydantic_to_json_schema

_client: openai.OpenAI | None = None
_client_lock = threading.Lock()


def _get_client(timeout: float) -> openai.OpenAI:
    """Module-level singleton for HTTP keep-alive (CTO MAJ-1)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = openai.OpenAI(timeout=timeout)
    return _client


def reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


def call_openai(
    *,
    model_class: type[MarkdownRenderable],
    prompt: str,
    system: str | None,
    llm_model: str,
    max_tokens: int,
    timeout_seconds: float,
    provider_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """OpenAI Chat Completions with json_schema strict mode.

    Returns: (parsed_dict, tokens_in, tokens_out)
    Raises: _ProviderParseFailure if response content isn't valid JSON.
            StructuredOutputProviderError on SDK errors.
    """
    schema = pydantic_to_json_schema(model_class)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = _get_client(timeout_seconds)
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": model_class.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
            max_tokens=max_tokens,
            **provider_kwargs,
        )
    except openai.OpenAIError as e:
        raise StructuredOutputProviderError(
            f"OpenAI API error: {e}",
            provider="openai",
            llm_model=llm_model,
            schema_name=model_class.__name__,
            underlying_exception=e,
            status_code=getattr(e, "status_code", None),
        ) from e

    tokens_in = response.usage.prompt_tokens if response.usage else 0
    tokens_out = response.usage.completion_tokens if response.usage else 0
    content = response.choices[0].message.content or ""
    try:
        return json.loads(content), tokens_in, tokens_out
    except json.JSONDecodeError as e:
        raise _ProviderParseFailure(
            raw_response=content[:2000],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason=f"json_decode: {e.msg}",
        ) from e
