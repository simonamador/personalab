"""Job specification and result types for the runtime layer.

Every execution unit is a *JobSpec* carrying a deterministic fingerprint
(job_id, seed, config_hash).  Results carry a monotonic *sequence_number*
so JSONL records can be sorted back into a stable order regardless of
async execution ordering.
"""

from __future__ import annotations

import hashlib
import json
import itertools
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from personalab.schemas.prompts import ImagePrompt, VideoPrompt

_global_seq = itertools.count()


def _next_seq() -> int:
    return next(_global_seq)


def compute_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation of *config*."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class JobSpec(BaseModel):
    """Immutable descriptor for a single generation job."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    seed: int = Field(default=0, description="Seed for reproducibility")
    config_hash: str = Field(default="", description="SHA-256 of the full job config snapshot")
    persona: str
    scenario_template: dict[str, Any]
    prompt: ImagePrompt | VideoPrompt
    asset_type: Literal["image", "video"]
    generation_id: str | None = None
    output_prefix: str = "gen"

    model_config = {"frozen": True}


class JobResult(BaseModel):
    """Outcome of a completed (or failed) job."""

    job_id: str
    seed: int = 0
    config_hash: str = ""
    sequence_number: int = Field(default_factory=_next_seq)
    asset_type: Literal["image", "video"] = "image"
    accepted: bool = False
    final_attempt: int = 0
    final_score: float = 0.0
    final_path: str | None = None
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: str = ""
    error: str | None = None
    attempts_summary: list[dict[str, Any]] = Field(default_factory=list)
