"""Tests for ReplicateAdapter with mocked Replicate SDK."""

import pytest

from personalab.llm.client import TextPart, LLMImageResponse


class TestReplicateAdapter:
    def test_parse_aspect_ratio(self):
        from personalab.llm.replicate_adapter import ReplicateAdapter
        assert ReplicateAdapter._parse_aspect_ratio("4:5") == (832, 1024)
        assert ReplicateAdapter._parse_aspect_ratio("9:16") == (576, 1024)
        assert ReplicateAdapter._parse_aspect_ratio("weird") == (1024, 1024)

    def test_generate_text_json_raises(self):
        from personalab.llm.replicate_adapter import ReplicateAdapter
        adapter = ReplicateAdapter.__new__(ReplicateAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_text_json(
                system_instruction="s", user_prompt="u",
                schema={}, use_search=False, model_name="m",
            )

    def test_generate_video_raises(self):
        from personalab.llm.replicate_adapter import ReplicateAdapter
        adapter = ReplicateAdapter.__new__(ReplicateAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_video(
                prompt="p", resolution="1080p", aspect_ratio="9:16", model_name="m",
            )

    def test_generate_image_with_mock(self, monkeypatch):
        """End-to-end with mocked replicate.Client.run and httpx."""
        from personalab.llm.replicate_adapter import ReplicateAdapter
        from unittest.mock import MagicMock, patch
        from types import SimpleNamespace

        adapter = ReplicateAdapter.__new__(ReplicateAdapter)
        mock_client = MagicMock()
        mock_client.run.return_value = ["https://example.com/image.png"]
        adapter._client = mock_client

        fake_resp = SimpleNamespace(content=b"PNG_BYTES", status_code=200)
        fake_resp.raise_for_status = lambda: None

        mock_http = MagicMock()
        mock_http.__enter__ = lambda self: mock_http
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.get.return_value = fake_resp

        with patch("personalab.llm.replicate_adapter.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_http
            resp = adapter.generate_image(
                parts=[TextPart(text="a cat")],
                aspect_ratio="1:1",
                model_name="stability-ai/sdxl:abc",
            )

        assert isinstance(resp, LLMImageResponse)
        assert resp.images == [b"PNG_BYTES"]
        mock_client.run.assert_called_once()
        call_args = mock_client.run.call_args
        assert call_args[0][0] == "stability-ai/sdxl:abc"
        assert call_args[1]["input"]["prompt"] == "a cat"
