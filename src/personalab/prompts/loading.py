"""Load personas and asset_scenarios with package defaults (data/examples/) + optional project overrides (paths.prompts)."""

from pathlib import Path
from typing import Any

from personalab.config import ProjectConfig
from personalab.config._yaml import load_yaml, load_package_yaml


def _load_package_default(*candidates: str) -> dict[str, Any]:
    """Load first existing YAML from package data (e.g. examples/personas.yml, prompts/personas.yml)."""
    for rel_path in candidates:
        try:
            data = load_package_yaml(rel_path)
            if data:
                return data
        except Exception:
            continue
    return {}


def load_personas(config: ProjectConfig) -> dict[str, Any]:
    """Load personas: package default from data/examples/personas.yml.
    User keys override or extend defaults."""
    default = _load_package_default("examples/personas.yml")
    prompts_dir = config.paths.get("prompts")
    if not prompts_dir:
        return default
    path = Path(prompts_dir) / "personas.yml"
    if not path.exists():
        return default
    user = load_yaml(path)
    return {**default, **user}


def load_asset_scenarios(config: ProjectConfig) -> dict[str, Any]:
    """Load asset_scenarios: package default from paths.prompts/asset_scenarios.yml.
    User keys override or extend defaults (e.g. image_scenarios, video_scenarios)."""
    default = _load_package_default("prompts/asset_scenarios.yml")
    prompts_dir = config.paths.get("prompts")
    if not prompts_dir:
        return default
    path = Path(prompts_dir) / "asset_scenarios.yml"
    if not path.exists():
        return default
    user = load_yaml(path)
    return {**default, **user}


def load_system_prompts(config: ProjectConfig) -> dict[str, Any]:
    """Load system_prompts: package default merged with optional project file at paths.prompts/system_prompts.yml."""
    default = load_package_yaml("prompts/system_prompts.yml")
    prompts_dir = config.paths.get("prompts")
    if not prompts_dir:
        return default
    path = Path(prompts_dir) / "system_prompts.yml"
    if not path.exists():
        return default
    user = load_yaml(path)
    return {**default, **user}


def load_anchor_templates(config: ProjectConfig) -> dict[str, Any]:
    """Load anchor templates (for reference image generation): package default from references/character_create.yml,
    merged with optional project file at paths.prompts/anchor_templates.yml or character_create.yml.
    Structure is keyed by template name (e.g. anchor_frontal_neutral). Character schema comes from Pydantic (Character)."""
    try:
        default = load_package_yaml("references/character_create.yml")
    except Exception:
        default = {}
    prompts_dir = config.paths.get("prompts")
    ref_templates = config.paths.get("reference_templates") or ""
    for base in (prompts_dir, ref_templates):
        if not base:
            continue
        path = Path(base) / "anchor_templates.yml"
        if path.exists():
            user = load_yaml(path)
            return {**default, **user}
        path = Path(base) / "character_create.yml"
        if path.exists():
            user = load_yaml(path)
            return {**default, **user}
    return default
