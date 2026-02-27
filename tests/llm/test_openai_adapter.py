"""Tests for OpenAIAdapter with mocked OpenAI SDK."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from personalab.llm.client import LLMTextResponse


class TestOpenAIAdapter:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("personalab.llm.openai_adapter.openai")
    def _make_adapter(self, mock_openai):
        """Helper: import and create adapter with mocked SDK."""
        # Re-import so the guarded import picks up the mock
        from personalab.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(api_key="test-key")
        return adapter, mock_openai

    def test_generate_text_json_parsed(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
            mock_openai = MagicMock()
            mock_choice = SimpleNamespace(
                message=SimpleNamespace(content='{"planned_content":[]}')
            )
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[mock_choice]
            )
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from personalab.llm.openai_adapter import OpenAIAdapter
                adapter = OpenAIAdapter.__new__(OpenAIAdapter)
                adapter._client = mock_openai.OpenAI(api_key="k")

                resp = adapter.generate_text_json(
                    system_instruction="sys",
                    user_prompt="user",
                    schema={},
                    use_search=False,
                    model_name="gpt-4o",
                )
                assert isinstance(resp, LLMTextResponse)
                assert resp.parsed == {"planned_content": []}
                assert resp.text == '{"planned_content":[]}'

    def test_generate_text_json_invalid_json(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
            mock_openai = MagicMock()
            mock_choice = SimpleNamespace(
                message=SimpleNamespace(content="not json")
            )
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[mock_choice]
            )
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from personalab.llm.openai_adapter import OpenAIAdapter
                adapter = OpenAIAdapter.__new__(OpenAIAdapter)
                adapter._client = mock_openai.OpenAI(api_key="k")

                resp = adapter.generate_text_json(
                    system_instruction="s",
                    user_prompt="u",
                    schema={},
                    use_search=False,
                    model_name="m",
                )
                assert resp.parsed is None
                assert resp.text == "not json"

    def test_generate_image_raises(self):
        from personalab.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_image(parts=[], aspect_ratio="1:1", model_name="m")

    def test_generate_video_raises(self):
        from personalab.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        with pytest.raises(NotImplementedError):
            adapter.generate_video(prompt="p", resolution="1080p", aspect_ratio="9:16", model_name="m")
