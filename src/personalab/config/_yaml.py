"""Minimal YAML loading (internal)."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict from the filesystem."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_package_yaml(relative_path: str) -> dict[str, Any]:
    """Load a YAML file from the package data (personalab/data/...)."""
    from importlib.resources import files
    content = (files("personalab") / "data" / relative_path).read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}
