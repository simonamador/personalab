"""Structured JSONL metrics logger for generation attempts.

Each line records a complete snapshot of one generation + evaluation cycle:
scores, decision, timing, prompt metadata.
"""

from __future__ import annotations

import itertools
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personalab.identity.schemas import CandidateResult, EvaluationResult

_global_seq = itertools.count()


class GenerationMetricsLogger:
    """Append-only JSONL logger. One file per persona under *output_dir*/{persona}/metrics.jsonl.

    Thread-safe: a ``threading.Lock`` serialises writes and a monotonic
    *sequence_number* is attached to each record so consumers can
    reconstruct insertion order even when written concurrently.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._lock = threading.Lock()

    def _persona_path(self, persona: str) -> Path:
        return self._output_dir / persona / "metrics.jsonl"

    def log(
        self,
        *,
        persona: str,
        generation_id: str | None = None,
        attempt: int,
        eval_result: EvaluationResult | CandidateResult | None = None,
        decision_action: str = "",
        generation_latency_ms: float | None = None,
        evaluation_latency_ms: float | None = None,
        prompt_hash: str = "",
        anchors_used: list[str] | None = None,
        model_name: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a structured metrics record and return it."""
        gen_id = generation_id or uuid.uuid4().hex[:12]

        scores = self._extract_scores(eval_result)

        record: dict[str, Any] = {
            "seq": next(_global_seq),
            "generation_id": gen_id,
            "persona": persona,
            "attempt": attempt,
            "scores": scores,
            "decision": decision_action,
            "timing": {
                "generation_latency_ms": generation_latency_ms,
                "evaluation_latency_ms": evaluation_latency_ms,
            },
            "prompt_hash": prompt_hash,
            "anchors_used": anchors_used or [],
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record["extra"] = extra

        path = self._persona_path(persona)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_log(self, persona: str, last_n: int | None = None) -> list[dict[str, Any]]:
        """Read back logged records for *persona*."""
        path = self._persona_path(persona)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        if last_n is not None:
            records = records[-last_n:]
        return records

    @staticmethod
    def _extract_scores(result: EvaluationResult | CandidateResult | None) -> dict[str, Any]:
        if result is None:
            return {}
        cand: CandidateResult
        if isinstance(result, EvaluationResult):
            if not result.candidates:
                return {}
            cand = result.candidates[0]
        else:
            cand = result

        scores: dict[str, Any] = {"composite_score": cand.composite_score}
        if cand.embedding:
            scores["embedding_similarity"] = cand.embedding.cosine_similarity
        if cand.geometric:
            scores["geometric_error"] = cand.geometric.normalized_error
        return scores
