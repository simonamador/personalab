"""Tests for IdeaGenerator with mocked LLMClient."""

import json
from pathlib import Path

import pytest
import yaml

from personalab.generation.idea_generator import IdeaGenerator
from personalab.llm.client import LLMTextResponse


SAMPLE_PLAN = {
    "planned_content": [
        {
            "id": "idea_1",
            "type": "image",
            "title": "Beach vibes",
            "scenario_details": {"scene": "beach", "outfit": "casual", "framing": "selfie"},
        }
    ]
}

SYSTEM_PROMPTS = {
    "content_strategist": {
        "system": "You are a strategist for ${NAME}. Vibe: ${VIBE}, location: ${LOCATION}, pillars: ${PILLARS}.",
        "user": "Generate 3 ideas.",
    }
}


class TestGenerate:
    def test_generate_returns_parsed_dict(self, fake_client, sample_config, sample_character):
        resp = LLMTextResponse(parsed=SAMPLE_PLAN)
        client = fake_client(text_json_response=resp)
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
        )
        result = ig.generate()
        assert "planned_content" in result
        assert len(result["planned_content"]) == 1
        assert client.calls[0]["method"] == "generate_text_json"
        assert client.calls[0]["model"] == "gemini-test"

    def test_generate_falls_back_to_text_json(self, fake_client, sample_config, sample_character):
        resp = LLMTextResponse(parsed=None, text=json.dumps(SAMPLE_PLAN))
        client = fake_client(text_json_response=resp)
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
        )
        result = ig.generate()
        assert "planned_content" in result

    def test_generate_with_character_dict(self, fake_client, sample_config):
        resp = LLMTextResponse(parsed=SAMPLE_PLAN)
        client = fake_client(text_json_response=resp)
        persona_dict = {
            "name": "Test",
            "vibe": "Chill",
            "location": "Lima",
            "content_pillars": ["Food"],
        }
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=persona_dict,
            system_prompts=SYSTEM_PROMPTS,
        )
        result = ig.generate()
        assert "planned_content" in result

    def test_raises_on_missing_system_prompt(self, fake_client, sample_config, sample_character):
        client = fake_client()
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts={},
        )
        with pytest.raises(KeyError, match="content_strategist"):
            ig.generate()

    def test_raises_on_missing_schema(self, fake_client, sample_config, sample_character):
        client = fake_client()
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
            schemas={},
        )
        with pytest.raises(ValueError, match="content_plan_schema"):
            ig.generate()

    def test_raises_on_missing_text_model(self, fake_client, sample_character):
        from personalab.config.loader import ProjectConfig
        no_model_config = ProjectConfig(raw={
            "paths": {"output": "."},
            "models": {},
            "generation": {"search": {"enabled": False}},
        })
        client = fake_client()
        ig = IdeaGenerator(
            client=client,
            config=no_model_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
        )
        with pytest.raises(ValueError, match="config.models.text"):
            ig.generate()


class TestSave:
    def test_save_writes_yaml(self, tmp_path, fake_client, sample_config, sample_character):
        resp = LLMTextResponse(parsed=SAMPLE_PLAN)
        client = fake_client(text_json_response=resp)
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
        )
        out_path = tmp_path / "plan.yml"
        ig.save(SAMPLE_PLAN, out_path)
        assert out_path.exists()

        loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert loaded["planned_content"][0]["title"] == "Beach vibes"

    def test_generate_and_save(self, tmp_path, fake_client, sample_config, sample_character):
        resp = LLMTextResponse(parsed=SAMPLE_PLAN)
        client = fake_client(text_json_response=resp)
        ig = IdeaGenerator(
            client=client,
            config=sample_config,
            persona=sample_character,
            system_prompts=SYSTEM_PROMPTS,
        )
        out_path = tmp_path / "sub" / "plan.yml"
        plan = ig.generate_and_save(out_path)
        assert out_path.exists()
        assert "planned_content" in plan
