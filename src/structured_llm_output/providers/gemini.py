"""Gemini provider — Vertex AI or Google AI Studio backend.

Gemini's structured-output mode uses `response_schema` + `response_mime_type:
application/json`, the equivalent of Anthropic's tool_use forced binding.
The model returns a JSON object that conforms to the schema; on rare parse
failures, _ProviderParseFailure triggers BB1's corrective-retry loop.

Backend selection (matches the official google-genai SDK convention):
- `GOOGLE_GENAI_USE_VERTEXAI=true` (+ `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_CLOUD_LOCATION`) → Vertex AI, Application Default Credentials
- otherwise `GOOGLE_API_KEY` → Google AI Studio direct API

Both backends share the same wire shape — the same provider function works
for either; only the singleton client differs.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from .._internal import _ProviderParseFailure
from ..exceptions import StructuredOutputProviderError
from ..media import MediaInput
from ..renderable import MarkdownRenderable
from ..schema_utils import pydantic_to_json_schema

_client: Any = None  # google.genai.Client — typed loosely so import is lazy
_client_lock = threading.Lock()


def _import_genai():
    try:
        from google import genai  # type: ignore
        return genai
    except ImportError as e:
        raise ImportError(
            "google-genai is required for the gemini provider. "
            "Install via: pip install 'structured-llm-output[gemini]'"
        ) from e


def _get_client(timeout: float) -> Any:
    """Module-level singleton client.

    `timeout` is currently advisory — google-genai's HTTP timeout is set on
    individual calls via `http_options`, not on the client. The argument is
    accepted to match the provider-function contract.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                genai = _import_genai()
                use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
                    "1", "true", "yes",
                )
                if use_vertex:
                    _client = genai.Client(
                        vertexai=True,
                        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
                    )
                else:
                    _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def reset_client() -> None:
    """Drop the singleton — for test isolation."""
    global _client
    with _client_lock:
        _client = None


def _strip_unsupported_schema_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's response_schema rejects a few JSON-Schema keywords that
    Pydantic emits but Gemini doesn't honor.

    Specifically: `additionalProperties`, `$defs`, `$ref` (Gemini wants the
    schema fully inlined), and `default`. Strips them recursively. The
    same approach is recommended by the google-genai cookbook.
    """
    if not isinstance(schema, dict):
        return schema
    drop = {"additionalProperties", "$schema", "default", "title"}
    cleaned: dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            cleaned[k] = _strip_unsupported_schema_fields(v)
        elif isinstance(v, list):
            cleaned[k] = [
                _strip_unsupported_schema_fields(x) if isinstance(x, dict) else x
                for x in v
            ]
        else:
            cleaned[k] = v
    return cleaned


def _build_contents(prompt: str, attachments: list[MediaInput] | None) -> Any:
    """Gemini `contents`: inline image/PDF Parts followed by the text prompt.

    With no attachments, returns the bare prompt string (unchanged wire shape).
    `Part.from_bytes` handles both images and PDFs via mime_type."""
    if not attachments:
        return prompt
    genai = _import_genai()
    parts = [
        genai.types.Part.from_bytes(data=bytes(att.data), mime_type=att.mime_type)
        for att in attachments
    ]
    return [*parts, prompt]


def call_gemini(
    *,
    model_class: type[MarkdownRenderable],
    prompt: str,
    system: str | None,
    llm_model: str,
    max_tokens: int,
    timeout_seconds: float,
    provider_kwargs: dict[str, Any],
    attachments: list[MediaInput] | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Gemini structured output via response_schema + JSON mime.

    Returns: (parsed_input_dict, tokens_in, tokens_out)
    Raises: _ProviderParseFailure on malformed JSON or non-object root.
            StructuredOutputProviderError on SDK errors (rate limit, 5xx, timeout).
    """
    schema = _strip_unsupported_schema_fields(pydantic_to_json_schema(model_class))
    client = _get_client(timeout_seconds)

    config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "max_output_tokens": max_tokens,
        # Enforce the per-call timeout on Gemini. google-genai's HTTP timeout is
        # request-level via http_options (milliseconds), NOT the client timeout —
        # without this the `timeout_seconds` contract is silently ignored.
        "http_options": {"timeout": int(timeout_seconds * 1000)},
    }
    if system:
        config["system_instruction"] = system
    # Caller-supplied overrides take precedence (e.g. temperature, top_p)
    config.update(provider_kwargs)

    contents = _build_contents(prompt, attachments)
    try:
        response = client.models.generate_content(
            model=llm_model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        # google.genai.errors.APIError + ClientError + ServerError live under
        # the google.genai module; fall through and re-raise anything that
        # doesn't look like an API error so we don't swallow programming bugs.
        if "google.genai" in type(e).__module__ or "google.api_core" in type(e).__module__:
            status_code = (
                getattr(e, "code", None)
                or getattr(e, "status_code", None)
            )
            raise StructuredOutputProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                llm_model=llm_model,
                schema_name=model_class.__name__,
                underlying_exception=e,
                status_code=status_code,
            ) from e
        raise

    usage = getattr(response, "usage_metadata", None)
    tokens_in = getattr(usage, "prompt_token_count", 0) or 0
    tokens_out = getattr(usage, "candidates_token_count", 0) or 0

    text = getattr(response, "text", None) or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise _ProviderParseFailure(
            raw_response=text[:2000],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason=f"json_decode: {e.msg}",
        ) from e

    if not isinstance(parsed, dict):
        raise _ProviderParseFailure(
            raw_response=text[:2000],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason=f"json_root_not_object: got {type(parsed).__name__}",
        )

    return parsed, tokens_in, tokens_out
