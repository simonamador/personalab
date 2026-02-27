"""Tests for config loader."""

from pathlib import Path

import pytest
from personalab.config import load_config, ProjectConfig


def test_project_config_properties():
    config = ProjectConfig(raw={
        "paths": {"output": "/out", "references": "/refs"},
        "models": {"text": "gemini-1", "image": "imagen-1"},
        "generation": {"search": {"enabled": True}},
    })
    assert config.paths["output"] == "/out"
    assert config.models["text"] == "gemini-1"
    assert config.generation["search"]["enabled"] is True


def test_load_config_default_from_package():
    """Default config uses workspace paths outside package."""
    config = load_config()
    assert isinstance(config, ProjectConfig)
    assert config.raw is not None
    assert "paths" in config.raw
    assert "prompts" in config.paths
    assert "references" in config.paths
    assert "output" in config.paths
    # Workspace dirs default to project-relative paths
    assert config.paths["references"] == "./references"
    assert config.paths["output"] == "./generated_content"
    assert config.paths["prompts"] == "./prompts"


def test_model_name_new_format():
    config = ProjectConfig(raw={
        "models": {
            "text": {"provider": "openai", "model_name": "gpt-4o"},
            "image": {"provider": "runware", "model_name": "runware:101@1"},
        }
    })
    assert config.model_name("text") == "gpt-4o"
    assert config.model_name("image") == "runware:101@1"
    assert config.model_name("video") == ""


def test_model_name_legacy_string():
    config = ProjectConfig(raw={
        "models": {"text": "gemini-1", "image": "imagen-1"}
    })
    assert config.model_name("text") == "gemini-1"
    assert config.model_name("image") == "imagen-1"


def test_provider_new_format():
    config = ProjectConfig(raw={
        "models": {
            "text": {"provider": "openai", "model_name": "gpt-4o"},
            "video": {"provider": "runway", "model_name": "gen3a_turbo"},
        }
    })
    assert config.provider("text") == "openai"
    assert config.provider("video") == "runway"
    assert config.provider("image") == "gemini"


def test_provider_legacy_string_defaults_to_gemini():
    config = ProjectConfig(raw={
        "models": {"text": "gemini-flash"}
    })
    assert config.provider("text") == "gemini"
