"""Extended tests for config loader: edge cases and the empty-path fix."""

import yaml
import pytest

from personalab.config import load_config, ProjectConfig


def test_load_config_with_explicit_path(tmp_path):
    cfg_file = tmp_path / "custom.yml"
    cfg_file.write_text(yaml.dump({
        "project": {"name": "Test"},
        "paths": {"output": "/custom/out"},
        "models": {"text": "gemini-test"},
    }), encoding="utf-8")
    config = load_config(cfg_file)
    assert config.paths["output"] == "/custom/out"
    # Empty-path fix: references and prompts get defaults
    assert config.paths["references"] == "./references"
    assert config.paths["prompts"] == "./prompts"


def test_load_config_fills_empty_string_paths(tmp_path):
    """Empty string paths should be replaced with defaults (regression test for the fix)."""
    cfg_file = tmp_path / "empty_paths.yml"
    cfg_file.write_text(yaml.dump({
        "paths": {"output": "", "references": "", "prompts": ""},
    }), encoding="utf-8")
    config = load_config(cfg_file)
    assert config.paths["output"] == "./generated_content"
    assert config.paths["references"] == "./references"
    assert config.paths["prompts"] == "./prompts"


def test_load_config_preserves_nonempty_paths(tmp_path):
    cfg_file = tmp_path / "custom_paths.yml"
    cfg_file.write_text(yaml.dump({
        "paths": {"output": "/my/out", "references": "/my/refs", "prompts": "/my/prompts"},
    }), encoding="utf-8")
    config = load_config(cfg_file)
    assert config.paths["output"] == "/my/out"
    assert config.paths["references"] == "/my/refs"
    assert config.paths["prompts"] == "/my/prompts"


def test_project_config_defaults_for_missing_sections():
    config = ProjectConfig(raw={})
    assert config.paths == {}
    assert config.models == {}
    assert config.generation == {}
