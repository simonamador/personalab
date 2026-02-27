"""Tests for LLM client abstractions (parts and response types)."""

import pytest

from personalab.llm.client import (
    TextPart, BytesPart, ContentPart,
    LLMTextResponse, LLMImageResponse, LLMVideoResponse,
)


class TestTextPart:
    def test_creation(self):
        p = TextPart(text="hello")
        assert p.text == "hello"

    def test_frozen(self):
        p = TextPart(text="immutable")
        with pytest.raises(AttributeError):
            p.text = "changed"

    def test_equality(self):
        assert TextPart(text="a") == TextPart(text="a")
        assert TextPart(text="a") != TextPart(text="b")


class TestBytesPart:
    def test_creation(self):
        p = BytesPart(data=b"\x89PNG", mime_type="image/png")
        assert p.data == b"\x89PNG"
        assert p.mime_type == "image/png"

    def test_frozen(self):
        p = BytesPart(data=b"x", mime_type="image/jpeg")
        with pytest.raises(AttributeError):
            p.mime_type = "other"

    def test_equality(self):
        a = BytesPart(data=b"same", mime_type="image/png")
        b = BytesPart(data=b"same", mime_type="image/png")
        assert a == b

    def test_different_data(self):
        a = BytesPart(data=b"a", mime_type="image/png")
        b = BytesPart(data=b"b", mime_type="image/png")
        assert a != b


class TestContentPartUnion:
    def test_text_is_content_part(self):
        p: ContentPart = TextPart(text="x")
        assert isinstance(p, TextPart)

    def test_bytes_is_content_part(self):
        p: ContentPart = BytesPart(data=b"x", mime_type="image/png")
        assert isinstance(p, BytesPart)


class TestLLMTextResponse:
    def test_defaults(self):
        r = LLMTextResponse()
        assert r.parsed is None
        assert r.text == ""

    def test_with_parsed(self):
        r = LLMTextResponse(parsed={"key": "val"}, text='{"key":"val"}')
        assert r.parsed == {"key": "val"}
        assert r.text == '{"key":"val"}'

    def test_frozen(self):
        r = LLMTextResponse(text="x")
        with pytest.raises(AttributeError):
            r.text = "y"


class TestLLMImageResponse:
    def test_defaults(self):
        r = LLMImageResponse()
        assert r.images == []

    def test_with_images(self):
        r = LLMImageResponse(images=[b"a", b"b"])
        assert len(r.images) == 2

    def test_frozen(self):
        r = LLMImageResponse(images=[b"x"])
        with pytest.raises(AttributeError):
            r.images = []


class TestLLMVideoResponse:
    def test_defaults(self):
        r = LLMVideoResponse()
        assert r.video_data is None
        assert r.operation_name == ""

    def test_with_data(self):
        r = LLMVideoResponse(video_data=b"vid", operation_name="op-1")
        assert r.video_data == b"vid"
        assert r.operation_name == "op-1"

    def test_frozen(self):
        r = LLMVideoResponse()
        with pytest.raises(AttributeError):
            r.operation_name = "x"
