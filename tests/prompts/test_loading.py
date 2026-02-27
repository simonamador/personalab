"""Tests for load_personas / load_asset_scenarios (package default + project merge)."""

import pytest
from personalab import load_config
from personalab.prompts import load_personas, load_asset_scenarios, load_system_prompts


def test_load_personas_returns_defaults_when_no_project_file():
    config = load_config()
    personas = load_personas(config)
    assert isinstance(personas, dict)
    assert "daniperez" in personas
    assert "name" in personas["daniperez"]
    assert "content_pillars" in personas["daniperez"]


def test_load_asset_scenarios_returns_defaults_when_no_project_file():
    config = load_config()
    scenarios = load_asset_scenarios(config)
    assert "image_scenarios" in scenarios
    assert "video_scenarios" in scenarios


def test_load_system_prompts_returns_defaults_when_no_project_file():
    config = load_config()
    prompts = load_system_prompts(config)
    assert "content_strategist" in prompts
    assert "system" in prompts["content_strategist"]
