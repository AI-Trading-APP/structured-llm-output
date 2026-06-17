"""Binary attachments (images / PDFs) for multimodal structured-output calls.

`call_structured(..., attachments=[MediaInput(...)])` sends one or more files
alongside the text prompt to a vision-capable model, while still binding the
response to a Pydantic schema. Used by EPI (portfolio screenshot/PDF
extraction) and any future document-understanding consumer.

Provider support:
  - anthropic : images (image block) + PDF (document block)
  - gemini    : images + PDF (inline Part)
  - openai    : images only (image_url); PDF raises ValueError
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

#: Image MIME types accepted by every vision-capable provider.
IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
)

#: PDF — supported by anthropic + gemini, rejected by the openai chat path.
PDF_MIME_TYPE = "application/pdf"

#: Everything `MediaInput` will accept.
SUPPORTED_MIME_TYPES = IMAGE_MIME_TYPES | {PDF_MIME_TYPE}


@dataclass(frozen=True)
class MediaInput:
    """A single binary attachment sent with the prompt.

    Args:
        mime_type: One of ``SUPPORTED_MIME_TYPES``.
        data:      Raw file bytes (not base64 — encoding is handled per provider).

    Raises:
        ValueError: unsupported ``mime_type`` or empty ``data``.
        TypeError:  ``data`` is not bytes.
    """

    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        if self.mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported media mime_type {self.mime_type!r}; "
                f"expected one of {sorted(SUPPORTED_MIME_TYPES)}"
            )
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("MediaInput.data must be bytes")
        if len(self.data) == 0:
            raise ValueError("MediaInput.data is empty")

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == PDF_MIME_TYPE

    def b64(self) -> str:
        """Standard base64 of the raw bytes (ASCII str)."""
        return base64.standard_b64encode(bytes(self.data)).decode("ascii")
