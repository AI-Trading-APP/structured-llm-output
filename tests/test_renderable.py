import pytest

from structured_llm_output import MarkdownRenderable


def test_base_to_markdown_raises():
    class Empty(MarkdownRenderable):
        x: int = 0

    with pytest.raises(NotImplementedError):
        Empty(x=1).to_markdown()


def test_subclass_round_trip(schemas, gd):
    """GD-SLO-010: ScreenerFilter markdown is stable and locked by exact-match."""
    scenario = gd("GD-SLO-010")
    instance = schemas["ScreenerFilter"](**scenario["input"]["instance"])
    assert instance.to_markdown() == scenario["expected_markdown"]
