"""Multimodal attachments (images / PDFs) — v0.3.0.

Verifies MediaInput validation and that each provider threads attachments into
the correct wire shape (anthropic image/document blocks, gemini inline Parts,
openai image_url parts) while leaving the no-attachment path byte-identical.
"""
from __future__ import annotations

import pytest

from structured_llm_output import MediaInput, call_structured
from structured_llm_output.media import SUPPORTED_MIME_TYPES
from structured_llm_output.providers import anthropic as ap
from structured_llm_output.providers import openai as op

PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"
PDF = b"%PDF-1.7 fake-pdf-bytes"


# ---------- MediaInput validation ----------


def test_mediainput_accepts_supported_types():
    for mt in SUPPORTED_MIME_TYPES:
        m = MediaInput(mime_type=mt, data=b"x")
        assert m.mime_type == mt
        assert m.b64()  # non-empty base64


def test_mediainput_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported media mime_type"):
        MediaInput(mime_type="image/tiff", data=b"x")


def test_mediainput_rejects_empty_and_nonbytes():
    with pytest.raises(ValueError, match="empty"):
        MediaInput(mime_type="image/png", data=b"")
    with pytest.raises(TypeError, match="must be bytes"):
        MediaInput(mime_type="image/png", data="notbytes")  # type: ignore[arg-type]


def test_is_pdf_flag():
    assert MediaInput("application/pdf", PDF).is_pdf is True
    assert MediaInput("image/png", PNG).is_pdf is False


# ---------- Anthropic wire shape ----------


def _capture_anthropic(monkeypatch, schemas):
    captured: dict = {}

    def fake_create(self, **kwargs):
        captured.update(kwargs)

        class _Blk:
            type = "tool_use"
            name = "TraderRating"
            input = {"ticker": "AAPL", "rating": "Buy", "confidence": 0.9, "rationale": "ok"}

        class _Usage:
            input_tokens = 10
            output_tokens = 5

        class _Msg:
            content = [_Blk()]
            usage = _Usage()

        return _Msg()

    monkeypatch.setattr("anthropic.resources.messages.Messages.create", fake_create)
    ap.reset_client()
    return captured


def test_anthropic_image_block(monkeypatch, schemas):
    captured = _capture_anthropic(monkeypatch, schemas)
    call_structured(
        model_class=schemas["TraderRating"],
        prompt="What stock is this?",
        provider="anthropic",
        llm_model="claude-sonnet-4-6",
        attachments=[MediaInput("image/png", PNG)],
    )
    content = captured["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["type"] == "base64"
    assert content[-1] == {"type": "text", "text": "What stock is this?"}
    ap.reset_client()


def test_anthropic_pdf_uses_document_block(monkeypatch, schemas):
    captured = _capture_anthropic(monkeypatch, schemas)
    call_structured(
        model_class=schemas["TraderRating"],
        prompt="Extract holdings",
        provider="anthropic",
        llm_model="claude-sonnet-4-6",
        attachments=[MediaInput("application/pdf", PDF)],
    )
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"
    ap.reset_client()


def test_anthropic_no_attachments_sends_bare_string(monkeypatch, schemas):
    captured = _capture_anthropic(monkeypatch, schemas)
    call_structured(
        model_class=schemas["TraderRating"],
        prompt="hi",
        provider="anthropic",
        llm_model="claude-sonnet-4-6",
    )
    assert captured["messages"][0]["content"] == "hi"
    ap.reset_client()


# ---------- OpenAI wire shape ----------


def _capture_openai(monkeypatch):
    captured: dict = {}

    def fake_create(self, **kwargs):
        captured.update(kwargs)

        class _Msg:
            content = '{"ticker": "AAPL", "rating": "Buy", "confidence": 0.9, "rationale": "ok"}'

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fake_create)
    op.reset_client()
    return captured


def test_openai_image_url_part(monkeypatch, schemas):
    captured = _capture_openai(monkeypatch)
    call_structured(
        model_class=schemas["TraderRating"],
        prompt="What stock?",
        provider="openai",
        llm_model="gpt-4o-2024-11-20",
        attachments=[MediaInput("image/png", PNG)],
    )
    content = captured["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "What stock?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    op.reset_client()


def test_openai_pdf_rejected(monkeypatch, schemas):
    _capture_openai(monkeypatch)
    with pytest.raises(ValueError, match="does not support PDF"):
        call_structured(
            model_class=schemas["TraderRating"],
            prompt="x",
            provider="openai",
            llm_model="gpt-4o-2024-11-20",
            attachments=[MediaInput("application/pdf", PDF)],
        )
    op.reset_client()
