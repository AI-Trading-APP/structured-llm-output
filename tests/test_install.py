def test_imports():
    from structured_llm_output import (
        MarkdownRenderable,
        StructuredOutputError,
        StructuredOutputProviderError,
        StructuredOutputValidationError,
        call_structured,
    )

    assert callable(call_structured)
    assert issubclass(StructuredOutputValidationError, StructuredOutputError)
    assert issubclass(StructuredOutputProviderError, StructuredOutputError)
    assert MarkdownRenderable.__module__.startswith("structured_llm_output")
