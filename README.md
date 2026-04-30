# structured-llm-output

Pydantic-bound provider-native structured output for LLM calls.

Building Block 1 (BB1) of the [TradingAgents Adoption Program](../../specs/tradingagents-adoption/program-roadmap.md).

> **Status**: Phase 3 PR-A (foundations) — see [tasks.md](../../specs/structured-llm-output/tasks.md) for the full Phase 3 slicing. Retry, ScreenerService migration, and full docs land in PR-B/C/D.

## Quick start

```python
from structured_llm_output import call_structured, MarkdownRenderable

class Rating(MarkdownRenderable):
    ticker: str
    score: float
    reason: str
    def to_markdown(self) -> str:
        return f"**{self.ticker}**: {self.score:.2f} — {self.reason}"

result: Rating = call_structured(
    model_class=Rating,
    prompt="Rate NVDA on a 0-1 scale",
    provider="anthropic",
    llm_model="claude-haiku-4-5-20251001",
)
print(result.to_markdown())
```

## Install (consumer service)

From a sibling repo (e.g., `ScreenerService/Dockerfile`):

```dockerfile
COPY ../AITradingAPP/shared-libs/structured-llm-output /tmp/sllm
RUN pip install -e /tmp/sllm
```

For local dev:

```bash
pip install -e ../AITradingAPP/shared-libs/structured-llm-output
```

## Run tests

```bash
cd shared-libs/structured-llm-output
pip install -e ".[dev]"
pytest
```

Tests use stubbed provider SDKs — no live LLM calls, no API keys required.
