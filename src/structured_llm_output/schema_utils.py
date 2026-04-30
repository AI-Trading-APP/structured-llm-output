from __future__ import annotations

from typing import Any

from .renderable import MarkdownRenderable


def pydantic_to_json_schema(model_class: type[MarkdownRenderable]) -> dict[str, Any]:
    """Pydantic v2 model → JSON Schema (draft 2020-12).

    Output is suitable for both Anthropic tool input_schema and
    OpenAI json_schema strict mode.
    """
    return model_class.model_json_schema()
