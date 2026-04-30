from __future__ import annotations

from typing import Any


class _ProviderParseFailure(Exception):
    """Internal — raised by providers when output cannot be turned into a dict.

    Caught by core.call_structured and converted to StructuredOutputValidationError
    so callers see one consistent public exception type.
    """

    def __init__(
        self,
        *,
        raw_response: Any,
        tokens_in: int = 0,
        tokens_out: int = 0,
        reason: str = "",
    ) -> None:
        super().__init__(reason or "provider parse failure")
        self.raw_response = raw_response
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.reason = reason
