"""Tests for GeminiPromptBuilder with mocked LLMClient."""

import json
from types import SimpleNamespace

import pytest

from personalab.prompts.builder import GeminiPromptBuilder


SYSTEM_PROMPTS = {
    "user_prompt_to_scenario": {
        "system": "Convert to scenario JSON.",
    }
}

SCHEMAS = {
    "scenario_details_schema": {"type": "object", "properties": {}},
}


class TestUserToImagePrompt:
    def test_returns_image_prompt_from_parsed(self, fake_client, sample_config):
        parsed = {"scene": "park", "outfit": "jeans", "framing": "selfie"}
        resp = SimpleNamespace(parsed=parsed)
        client = fake_client(text_json_response=resp)
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas=SCHEMAS,
        )
        prompt = builder.user_to_image_prompt("walking in a park")
        assert prompt.scene_description == "park"
        assert prompt.clothing_details == "jeans"
        assert prompt.shot_type == "selfie"

    def test_falls_back_to_text_json(self, fake_client, sample_config):
        parsed = {"scene": "cafe", "outfit": "dress", "framing": "candid"}
        resp = SimpleNamespace(parsed=None, text=json.dumps(parsed))
        client = fake_client(text_json_response=resp)
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas=SCHEMAS,
        )
        prompt = builder.user_to_image_prompt("coffee date")
        assert prompt.scene_description == "cafe"

    def test_raises_on_invalid_json(self, fake_client, sample_config):
        resp = SimpleNamespace(parsed=None, text="not json!")
        client = fake_client(text_json_response=resp)
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas=SCHEMAS,
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            builder.user_to_image_prompt("test")

    def test_raises_on_non_dict_parsed(self, fake_client, sample_config):
        resp = SimpleNamespace(parsed=None, text=json.dumps(["a", "list"]))
        client = fake_client(text_json_response=resp)
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas=SCHEMAS,
        )
        with pytest.raises(TypeError, match="Expected dict"):
            builder.user_to_image_prompt("test")

    def test_raises_on_missing_system_prompt(self, fake_client, sample_config):
        client = fake_client()
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts={}, schemas=SCHEMAS,
        )
        with pytest.raises(ValueError, match="user_prompt_to_scenario"):
            builder.user_to_image_prompt("test")

    def test_raises_on_missing_schema(self, fake_client, sample_config):
        client = fake_client()
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas={},
        )
        with pytest.raises(ValueError, match="scenario_details_schema"):
            builder.user_to_image_prompt("test")

    def test_subject_description_from_context(self, fake_client, sample_config):
        parsed = {"scene": "park", "outfit": "jeans", "framing": "selfie"}
        resp = SimpleNamespace(parsed=parsed)
        client = fake_client(text_json_response=resp)
        builder = GeminiPromptBuilder(
            client=client, config=sample_config,
            system_prompts=SYSTEM_PROMPTS, schemas=SCHEMAS,
        )
        prompt = builder.user_to_image_prompt("test", context={"subject_description": "tall woman"})
        assert prompt.subject_description == "tall woman"
