"""Longitudinal drift tracking per persona.

Appends evaluation scores to a per-persona JSONL timeline and provides
rolling summary statistics (mean, std, trend).
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

from personalab.identity.schemas import (
    CandidateResult,
    DriftRecord,
    DriftSummary,
    EvaluationResult,
)


class DriftTracker:
    """Per-persona append-only JSONL timeline of identity scores (thread-safe)."""

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._lock = threading.Lock()

    def _persona_path(self, persona: str) -> Path:
        return self._log_dir / f"{persona}_drift.jsonl"

    def record(
        self,
        persona: str,
        eval_result: EvaluationResult | CandidateResult,
        generation_id: str = "",
        decision_action: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DriftRecord:
        """Append a drift record for *persona*. Returns the written record."""
        if isinstance(eval_result, EvaluationResult):
            cand = eval_result.first()
        else:
            cand = eval_result

        rec = DriftRecord(
            generation_id=generation_id,
            persona=persona,
            composite_score=cand.composite_score,
            embedding_similarity=cand.embedding.cosine_similarity if cand.embedding else None,
            geometric_error=cand.geometric.normalized_error if cand.geometric else None,
            decision_action=decision_action,
            metadata=metadata or {},
        )
        self._append(persona, rec)
        return rec

    def history(self, persona: str, window: int | None = None) -> list[DriftRecord]:
        """Return the last *window* drift records (all if window is None)."""
        path = self._persona_path(persona)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        records = [DriftRecord(**json.loads(line)) for line in lines if line.strip()]
        if window is not None:
            records = records[-window:]
        return records

    def summary(self, persona: str, window: int = 20) -> DriftSummary:
        """Compute rolling stats over the last *window* records."""
        records = self.history(persona, window=window)
        if not records:
            return DriftSummary(persona=persona, window_size=window)

        scores = [r.composite_score for r in records]
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)

        trend = _compute_trend(scores)

        return DriftSummary(
            persona=persona,
            window_size=window,
            count=n,
            mean_score=round(mean, 6),
            std_score=round(std, 6),
            min_score=round(min(scores), 6),
            max_score=round(max(scores), 6),
            trend=trend,
        )

    def _append(self, persona: str, record: DriftRecord) -> None:
        path = self._persona_path(persona)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")


def _compute_trend(scores: list[float], min_samples: int = 3) -> str:
    """Simple linear trend: positive slope = improving, negative = degrading."""
    if len(scores) < min_samples:
        return "stable"
    n = len(scores)
    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n
    num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return "stable"
    slope = num / den
    if slope > 0.005:
        return "improving"
    if slope < -0.005:
        return "degrading"
    return "stable"
