"""Tests for RunwareAdapter with mocked Runware SDK."""

import pytest

from personalab.llm.client import TextPart, LLMImageResponse


class TestRunwareAdapter:
    def test_parse_aspect_ratio(self):
        from personalab.llm.runware_adapter import RunwareAdapter
        assert RunwareAdapter._parse_aspect_ratio("4:5") == (832, 1024)
        assert RunwareAdapter._parse_aspect_ratio("1:1") == (1024, 1024)
        assert RunwareAdapter._parse_aspect_ratio("unknown") == (1024, 1024)

    def test_generate_text_json_raises(self):
        from personalab.llm.runware_adapter import RunwareAdapter
        adapter = RunwareAdapter.__new__(RunwareAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_text_json(
                system_instruction="s", user_prompt="u",
                schema={}, use_search=False, model_name="m",
            )

    def test_generate_video_raises(self):
        from personalab.llm.runware_adapter import RunwareAdapter
        adapter = RunwareAdapter.__new__(RunwareAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_video(
                prompt="p", resolution="1080p", aspect_ratio="9:16", model_name="m",
            )

    def test_prompt_extraction_from_parts(self):
        """Verify that text parts are concatenated into the prompt string."""
        from personalab.llm.runware_adapter import RunwareAdapter
        from personalab.llm.client import BytesPart

        adapter = RunwareAdapter.__new__(RunwareAdapter)
        adapter._api_key = "fake"

        parts = [
            TextPart(text="identity lock"),
            BytesPart(data=b"img", mime_type="image/png"),
            TextPart(text="generate a photo"),
        ]
        prompt_pieces = []
        for p in parts:
            if isinstance(p, TextPart):
                prompt_pieces.append(p.text)
        prompt = "\n".join(prompt_pieces)
        assert "identity lock" in prompt
        assert "generate a photo" in prompt
