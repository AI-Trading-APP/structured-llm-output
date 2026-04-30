from __future__ import annotations

from typing import Any


class StructuredOutputError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        llm_model: str,
        schema_name: str,
        raw_response: Any = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.llm_model = llm_model
        self.schema_name = schema_name
        self.raw_response = raw_response


class StructuredOutputValidationError(StructuredOutputError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        llm_model: str,
        schema_name: str,
        raw_response: Any,
        validation_errors: list[dict[str, Any]],
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            llm_model=llm_model,
            schema_name=schema_name,
            raw_response=raw_response,
        )
        self.validation_errors = validation_errors
        self.attempts = attempts or []


class StructuredOutputProviderError(StructuredOutputError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        llm_model: str,
        schema_name: str,
        underlying_exception: Exception,
        status_code: int | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            llm_model=llm_model,
            schema_name=schema_name,
            raw_response=None,
        )
        self.underlying_exception = underlying_exception
        self.status_code = status_code
