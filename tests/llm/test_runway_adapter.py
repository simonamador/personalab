"""Tests for RunwayAdapter with mocked RunwayML SDK."""

import pytest

from personalab.llm.client import LLMVideoResponse


class TestRunwayAdapter:
    def test_generate_text_json_raises(self):
        from personalab.llm.runway_adapter import RunwayAdapter
        adapter = RunwayAdapter.__new__(RunwayAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_text_json(
                system_instruction="s", user_prompt="u",
                schema={}, use_search=False, model_name="m",
            )

    def test_generate_image_raises(self):
        from personalab.llm.runway_adapter import RunwayAdapter
        adapter = RunwayAdapter.__new__(RunwayAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_image(parts=[], aspect_ratio="1:1", model_name="m")

    def test_generate_video_with_mock(self, monkeypatch):
        """End-to-end with mocked RunwayML client and httpx."""
        from personalab.llm.runway_adapter import RunwayAdapter
        from unittest.mock import MagicMock, patch
        from types import SimpleNamespace

        adapter = RunwayAdapter.__new__(RunwayAdapter)
        mock_client = MagicMock()
        mock_client.image_to_video.create.return_value = SimpleNamespace(id="task-42")
        mock_client.tasks.retrieve.return_value = SimpleNamespace(
            status="SUCCEEDED",
            output=["https://example.com/video.mp4"],
        )
        adapter._client = mock_client

        fake_resp = SimpleNamespace(content=b"VIDEO_DATA", status_code=200)
        fake_resp.raise_for_status = lambda: None

        mock_http = MagicMock()
        mock_http.__enter__ = lambda self: mock_http
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.get.return_value = fake_resp

        with patch("personalab.llm.runway_adapter.httpx") as mock_httpx, \
             patch("personalab.llm.runway_adapter.time") as mock_time:
            mock_httpx.Client.return_value = mock_http
            mock_time.sleep = MagicMock()

            resp = adapter.generate_video(
                prompt="A sunset timelapse",
                resolution="1080p",
                aspect_ratio="16:9",
                model_name="gen3a_turbo",
            )

        assert isinstance(resp, LLMVideoResponse)
        assert resp.video_data == b"VIDEO_DATA"
        assert resp.operation_name == "task-42"
        mock_client.image_to_video.create.assert_called_once()
