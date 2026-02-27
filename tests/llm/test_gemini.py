"""Tests for GeminiAdapter part conversion (unit-testable without API key)."""

import pytest

from personalab.llm.client import TextPart, BytesPart
from personalab.llm.gemini import GeminiAdapter


class TestToGenaiParts:
    """Test the static _to_genai_parts converter without needing an API key."""

    def test_text_part_conversion(self):
        parts = GeminiAdapter._to_genai_parts([TextPart(text="hello")])
        assert len(parts) == 1
        assert parts[0].text == "hello"

    def test_bytes_part_conversion(self):
        raw = b"\x89PNG\r\n\x1a\n"
        parts = GeminiAdapter._to_genai_parts([BytesPart(data=raw, mime_type="image/png")])
        assert len(parts) == 1
        assert parts[0].inline_data.data == raw
        assert parts[0].inline_data.mime_type == "image/png"

    def test_mixed_parts(self):
        parts = GeminiAdapter._to_genai_parts([
            TextPart(text="identity lock"),
            BytesPart(data=b"img", mime_type="image/jpeg"),
            TextPart(text="generate"),
        ])
        assert len(parts) == 3
        assert parts[0].text == "identity lock"
        assert parts[1].inline_data.data == b"img"
        assert parts[2].text == "generate"

    def test_empty_list(self):
        assert GeminiAdapter._to_genai_parts([]) == []

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Unknown part type"):
            GeminiAdapter._to_genai_parts(["not a part"])
