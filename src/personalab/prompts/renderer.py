"""Recursive template rendering for prompt dicts."""

import json
from typing import Any


def _substitute(text: str, variables: dict[str, str]) -> str:
    """Replace ${VAR} placeholders in a single string."""
    out = text
    for k, v in variables.items():
        out = out.replace(f"${{{k}}}", v)
    return out


def render_string(template: str, variables: dict[str, str]) -> str:
    """Substitute ${VAR} placeholders in a single string template."""
    return _substitute(template, variables)


def render_prompt(template: dict[str, Any], variables: dict[str, str]) -> dict[str, Any]:
    """Recursively substitute ${VAR} inside a dict template."""
    def _render(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _render(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_render(x) for x in obj]
        if isinstance(obj, str):
            return _substitute(obj, variables)
        return obj
    return _render(template)


def prompt_to_str(prompt_dict: dict[str, Any]) -> str:
    """Serialize prompt dict to stable string for model input."""
    return json.dumps(prompt_dict, ensure_ascii=False, sort_keys=True, indent=None, separators=(",", ":"))
