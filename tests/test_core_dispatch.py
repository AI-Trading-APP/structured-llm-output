import pytest
from pydantic import BaseModel

from structured_llm_output import MarkdownRenderable, call_structured


def test_gd_slo_012_unknown_provider(schemas):
    """GD-SLO-012: provider must be one of the supported names.

    v0.2 added gemini, so the rejected example here is a clearly fake provider.
    """
    with pytest.raises(ValueError, match="provider must be one of"):
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt="x",
            provider="cohere",  # type: ignore[arg-type]
            llm_model="command-r",
        )


def test_non_renderable_class_raises():
    """REQ-SLO-005: model_class must extend MarkdownRenderable."""

    class NotRenderable(BaseModel):
        x: int = 0

    with pytest.raises(TypeError, match="MarkdownRenderable"):
        call_structured(
            model_class=NotRenderable,  # type: ignore[arg-type]
            prompt="x",
            provider="anthropic",
            llm_model="claude-haiku-4-5-20251001",
        )


def test_subclass_without_to_markdown_override_raises():
    """MIN-1: missing override surfaces at call time, not at runtime in production."""

    class Forgot(MarkdownRenderable):
        x: int = 0

    with pytest.raises(TypeError, match="to_markdown"):
        call_structured(
            model_class=Forgot,
            prompt="x",
            provider="anthropic",
            llm_model="claude-haiku-4-5-20251001",
        )


def test_provider_kwargs_reserved_keys_rejected(schemas):
    """MIN-2: reserved provider kwargs rejected with clear error."""
    with pytest.raises(ValueError, match="reserved keys"):
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt="x",
            provider="anthropic",
            llm_model="claude-haiku-4-5-20251001",
            provider_kwargs={"tools": []},
        )

    with pytest.raises(ValueError, match="reserved keys"):
        call_structured(
            model_class=schemas["ScreenerFilter"],
            prompt="x",
            provider="openai",
            llm_model="gpt-4o-2024-11-20",
            provider_kwargs={"response_format": {}},
        )
