from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Provider SDKs validate API key presence at client construction. Tests stub
# the actual network calls but the constructor still needs a value to be set.
os.environ.setdefault("OPENAI_API_KEY", "test-fake-key-not-used")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-fake-key-not-used")

import pytest  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from structured_llm_output import MarkdownRenderable  # noqa: E402
from structured_llm_output.providers import anthropic as ap  # noqa: E402
from structured_llm_output.providers import openai as op  # noqa: E402


_GOLDEN_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "specs"
    / "structured-llm-output"
    / "golden-data.json"
)


@pytest.fixture(scope="session")
def golden_data() -> dict[str, Any]:
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def gd(golden_data):
    by_id = {s["id"]: s for s in golden_data["scenarios"]}

    def _get(scenario_id: str) -> dict[str, Any]:
        return by_id[scenario_id]

    return _get


# ---------- Test schemas (extend MarkdownRenderable) ----------


class ScreenerFilter(MarkdownRenderable):
    market_cap_min: Optional[int] = None
    market_cap_max: Optional[int] = None
    sector: Optional[list[str]] = None
    pe_max: Optional[float] = None
    pe_min: Optional[float] = None
    exchange: Optional[list[str]] = None

    def to_markdown(self) -> str:
        lines = ["**Filter**"]
        if self.sector:
            lines.append(f"- Sector: {', '.join(self.sector)}")
        if self.market_cap_max:
            lines.append(f"- Market cap: ≤ ${self.market_cap_max / 1e9:.1f}B")
        if self.pe_max:
            lines.append(f"- P/E: ≤ {self.pe_max}")
        return "\n".join(lines)


class Rating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderRating(MarkdownRenderable):
    ticker: str
    rating: Rating
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    def to_markdown(self) -> str:
        return f"**{self.ticker}**: {self.rating.value} ({self.confidence:.0%}) — {self.rationale}"


class ResearchPlanActions(BaseModel):
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_pct: float = Field(ge=0.0, le=1.0)


class ResearchPlan(MarkdownRenderable):
    ticker: str
    rating: Rating
    thesis: str
    risks: list[str]
    actions: ResearchPlanActions

    def to_markdown(self) -> str:
        risks_md = "\n".join(f"  - {r}" for r in self.risks)
        return (
            f"# {self.ticker} — {self.rating.value}\n\n"
            f"**Thesis:** {self.thesis}\n\n"
            f"**Risks:**\n{risks_md}"
        )


@pytest.fixture(scope="session")
def schemas():
    return {
        "ScreenerFilter": ScreenerFilter,
        "TraderRating": TraderRating,
        "ResearchPlan": ResearchPlan,
    }


# ---------- Anthropic SDK stubs ----------


class _MockBlock:
    def __init__(self, type, name=None, input=None, text=None):
        self.type = type
        self.name = name
        self.input = input
        self.text = text


class _MockAnthropicUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicMessage:
    def __init__(self, content_blocks, usage):
        self.content = content_blocks
        self.usage = usage

    def model_dump(self):
        return {
            "content": [
                {"type": b.type, "name": b.name, "input": b.input, "text": b.text}
                for b in self.content
            ],
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
        }


def _build_anthropic_msg(scenario_response: dict[str, Any]) -> _MockAnthropicMessage:
    block = _MockBlock(
        type=scenario_response["type"],
        name=scenario_response.get("name"),
        input=scenario_response.get("input"),
        text=scenario_response.get("text"),
    )
    return _MockAnthropicMessage([block], _MockAnthropicUsage())


@pytest.fixture
def stub_anthropic(monkeypatch):
    """Set state['responses'] to a list of scenario_response dicts (one per call).

    For backward-compat with PR-A tests, also accepts state['response'] (single dict).
    For exception-injection, set state['exception'] (raised on every call).
    """
    state: dict[str, Any] = {"responses": [], "response": None, "exception": None}
    counter = {"i": 0}

    def fake_create(self, **kwargs):
        if state["exception"]:
            raise state["exception"]
        responses = state["responses"] or ([state["response"]] if state["response"] else [])
        i = counter["i"]
        counter["i"] += 1
        if i >= len(responses):
            raise IndexError(
                f"stub_anthropic: test made call #{i + 1} but only stubbed {len(responses)} responses"
            )
        return _build_anthropic_msg(responses[i])

    monkeypatch.setattr("anthropic.resources.messages.Messages.create", fake_create)
    ap.reset_client()
    yield state
    ap.reset_client()


# ---------- OpenAI SDK stubs ----------


class _MockOAIMessage:
    def __init__(self, content):
        self.content = content


class _MockOAIChoice:
    def __init__(self, content):
        self.message = _MockOAIMessage(content)


class _MockOAIUsage:
    def __init__(self, prompt_tokens=80, completion_tokens=40):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockOAIResponse:
    def __init__(self, content):
        self.choices = [_MockOAIChoice(content)]
        self.usage = _MockOAIUsage()


@pytest.fixture
def stub_openai(monkeypatch):
    """Set state['contents'] to a list of response content strings (one per call).

    For backward-compat with PR-A tests, also accepts state['content'] (single string).
    For exception-injection, set state['exception'].
    """
    state: dict[str, Any] = {"contents": [], "content": None, "exception": None}
    counter = {"i": 0}

    def fake_create(self, **kwargs):
        if state["exception"]:
            raise state["exception"]
        contents = state["contents"] or (
            [state["content"]] if state["content"] is not None else []
        )
        i = counter["i"]
        counter["i"] += 1
        if i >= len(contents):
            raise IndexError(
                f"stub_openai: test made call #{i + 1} but only stubbed {len(contents)} responses"
            )
        return _MockOAIResponse(contents[i])

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    op.reset_client()
    yield state
    op.reset_client()


# ---------- OTel test exporter ----------


@pytest.fixture
def otel_exporter():
    """In-memory OTel span exporter for test assertions.

    OTel SDK's set_tracer_provider only takes effect on the first call per process,
    so we install the provider directly via the module's internal slot. This is
    test-only and acceptable; production consumers initialize OTel normally.
    """
    from opentelemetry import trace as trace_mod
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Direct slot mutation — set_tracer_provider warns + ignores after first call.
    trace_mod._TRACER_PROVIDER = provider
    yield exporter
    exporter.clear()
