"""Load project config from YAML with a typed view."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from personalab.config._yaml import load_yaml, load_package_yaml


def get_package_data_root() -> Path:
    """Path to personalab/data/ (prompts, config, references). Works from source or installed package."""
    return Path(__file__).resolve().parent.parent / "data"


def get_reference_templates_root(config: "ProjectConfig") -> Path:
    """Path to reference templates (e.g. character_create.yml).
    Uses package data unless config.paths.reference_templates is set."""
    p = config.raw.get("paths", {}).get("reference_templates") or ""
    if p:
        return Path(p)
    return get_package_data_root() / "references"


_RUNTIME_DEFAULTS: dict[str, Any] = {
    "mode": "sync",
    "max_concurrent_jobs": 5,
    "eval_pool_workers": 2,
    "concurrency": {
        "gemini": 3,
        "openai": 5,
        "runware": 5,
        "replicate": 3,
        "runway": 2,
    },
    "rate_limits": {
        "gemini": 5.0,
        "openai": 10.0,
        "runware": 10.0,
        "replicate": 5.0,
        "runway": 2.0,
    },
    "retry": {
        "backoff_base": 1.0,
        "backoff_max": 30.0,
        "backoff_factor": 2.0,
    },
}


_EVALUATION_DEFAULTS: dict[str, Any] = {
    "embedding": {
        "backend": "arcface",
        "model_name": "buffalo_l",
        "similarity_threshold": 0.55,
    },
    "geometric": {
        "enabled": True,
        "max_normalized_error": 0.08,
    },
    "scoring": {
        "weights": {"embedding": 0.7, "geometric": 0.3},
        "accept_threshold": 0.55,
        "retry_threshold": 0.35,
        "policy_mode": "composite_only",
    },
    "retry": {
        "max_attempts": 3,
    },
    "drift": {
        "enabled": True,
        "log_dir": "./generated_content/drift",
    },
    "quality": {
        "enabled": True,
        "sharpness": {
            "min_ratio": 0.4,
        },
        "illumination": {
            "max_histogram_distance": 0.35,
            "method": "bhattacharyya",
        },
        "post_processing": {
            "auto_sharpen": False,
            "auto_super_res": False,
            "max_embedding_std": 0.06,
            "max_geometric_variance": 0.002,
        },
    },
}


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *defaults* (overrides win)."""
    merged = dict(defaults)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


class ProjectConfig(BaseModel):
    """Typed view over config YAML for stable access."""

    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def models(self) -> dict[str, Any]:
        return self.raw.get("models", {})

    def model_name(self, modality: str) -> str:
        """Return the model name for a modality (text / image / video).
        Supports both the new ``{provider, model_name}`` dict format and the
        legacy plain-string format for backward compatibility."""
        entry = self.models.get(modality)
        if entry is None:
            return ""
        if isinstance(entry, str):
            return entry
        return entry.get("model_name", "")

    def provider(self, modality: str) -> str:
        """Return the provider key for a modality (e.g. ``"gemini"``, ``"openai"``).
        Defaults to ``"gemini"`` when using the legacy plain-string config."""
        entry = self.models.get(modality)
        if entry is None:
            return "gemini"
        if isinstance(entry, str):
            return "gemini"
        return entry.get("provider", "gemini")

    @property
    def generation(self) -> dict[str, Any]:
        return self.raw.get("generation", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        """Evaluation config with defaults filled for missing keys."""
        user = self.raw.get("evaluation", {})
        return _deep_merge(_EVALUATION_DEFAULTS, user)

    @property
    def runtime(self) -> dict[str, Any]:
        """Runtime config (mode, concurrency, rate limits, pool sizes)."""
        user = self.raw.get("runtime", {})
        return _deep_merge(_RUNTIME_DEFAULTS, user)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load project config. If path is None, load default from package data.
    Default paths are workspace dirs outside the package:
    - references: where anchors are stored (e.g. ./references)
    - output: generated content (e.g. ./generated_content)
    - prompts: optional project folder for user overrides (e.g. ./prompts)
    """
    if path is None:
        raw = load_package_yaml("config/config.yml")
    else:
        raw = load_yaml(path)
    raw.setdefault("paths", {})
    defaults = {"references": "./references", "output": "./generated_content", "prompts": "./prompts"}
    for key, default in defaults.items():
        if not raw["paths"].get(key):
            raw["paths"][key] = default
    return ProjectConfig(raw=raw)
