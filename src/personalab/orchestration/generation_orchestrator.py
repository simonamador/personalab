"""Reusable generation-evaluate-retry orchestrator.

Encapsulates the loop: generate image -> evaluate identity -> policy decision -> retry with patches.
Previously this logic lived only in ``scripts/auto_generate_image.py``.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from personalab.generation.image_generator import ImageGenerator
from personalab.identity.evaluator import IdentityEvaluator
from personalab.identity.identity_policy import Decision, IdentityPolicy
from personalab.identity.drift import DriftTracker
from personalab.identity.schemas import EvaluationResult
from personalab.observability.metrics_logger import GenerationMetricsLogger
from personalab.schemas import ImageGenMeta, ImagePrompt

logger = logging.getLogger(__name__)


class GenerationAttempt(BaseModel):
    """Record of a single generation attempt within the orchestrator loop."""

    attempt: int
    image_path: str | None = None
    image_bytes_size: int = 0
    eval_result: EvaluationResult | None = None
    decision: Decision | None = None
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class OrchestrationResult(BaseModel):
    """Final result of the orchestration loop."""

    persona: str
    generation_id: str
    accepted: bool = False
    final_attempt: int = 0
    final_image_path: str | None = None
    final_score: float = 0.0
    attempts: list[GenerationAttempt] = Field(default_factory=list)


class GenerationOrchestrator:
    """Generate -> evaluate -> policy -> retry loop.

    All dependencies are injected; no global state.
    """

    def __init__(
        self,
        *,
        image_generator: ImageGenerator,
        evaluator: IdentityEvaluator,
        policy: IdentityPolicy,
        drift_tracker: DriftTracker | None = None,
        metrics_logger: GenerationMetricsLogger | None = None,
        output_dir: str | Path = "./generated_content",
        references_root: str | Path = "./references",
    ) -> None:
        self._img_gen = image_generator
        self._evaluator = evaluator
        self._policy = policy
        self._drift = drift_tracker
        self._metrics = metrics_logger
        self._output_dir = Path(output_dir)
        self._references_root = Path(references_root)

    def run(
        self,
        *,
        persona: str,
        scenario_template: dict[str, Any],
        image_prompt: ImagePrompt,
        generation_id: str | None = None,
        output_prefix: str = "gen",
    ) -> OrchestrationResult:
        """Execute the full generation-evaluation loop.

        Returns an ``OrchestrationResult`` with all attempt records.
        """
        gen_id = generation_id or uuid.uuid4().hex[:12]
        anchor_path = self._resolve_anchor(persona)

        result = OrchestrationResult(persona=persona, generation_id=gen_id)
        attempt = 0

        while attempt < self._policy.max_attempts:
            attempt += 1

            t0 = time.perf_counter()
            img_bytes, meta, prompt_used = self._img_gen.generate_asset_image(
                persona_name=persona,
                scenario_template=scenario_template,
                image_prompt=image_prompt,
            )
            gen_ms = (time.perf_counter() - t0) * 1000

            rec = GenerationAttempt(attempt=attempt, generation_ms=gen_ms)
            rec.meta = meta.model_dump() if isinstance(meta, ImageGenMeta) else {}

            if not img_bytes:
                logger.warning("Attempt %d: generation returned no bytes", attempt)
                result.attempts.append(rec)
                continue

            out_path = self._output_dir / persona / f"{output_prefix}_{attempt}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)
            rec.image_path = str(out_path)
            rec.image_bytes_size = len(img_bytes)

            if anchor_path:
                t1 = time.perf_counter()
                eval_result = self._evaluator.evaluate(str(anchor_path), [str(out_path)])
                eval_ms = (time.perf_counter() - t1) * 1000
                rec.eval_result = eval_result
                rec.evaluation_ms = eval_ms

                decision = self._policy.decide(attempt, eval_result)
            else:
                eval_result = None
                eval_ms = 0.0
                decision = Decision(action="ACCEPT", notes="no_anchor_available", score=1.0)

            rec.decision = decision
            result.attempts.append(rec)

            self._log_metrics(
                persona=persona, gen_id=gen_id, attempt=attempt,
                eval_result=eval_result, decision=decision,
                gen_ms=gen_ms, eval_ms=eval_ms, meta=meta,
            )

            if decision.action == "ACCEPT":
                result.accepted = True
                result.final_attempt = attempt
                result.final_image_path = str(out_path)
                result.final_score = decision.score
                logger.info("Accepted at attempt %d (score=%.4f)", attempt, decision.score)
                break
            elif decision.action == "RETRY":
                logger.info(
                    "Retry (attempt %d, score=%.4f): %s",
                    attempt, decision.score, decision.notes,
                )
                image_prompt.policy_overrides = decision.patches.get("policy_overrides")
            else:
                result.final_attempt = attempt
                result.final_score = decision.score
                logger.info("Rejected final (attempt %d, score=%.4f)", attempt, decision.score)
                break

        if not result.accepted and result.final_attempt == 0:
            result.final_attempt = attempt

        return result

    def _resolve_anchor(self, persona: str) -> Path | None:
        anchor = self._references_root / persona / "anchors" / f"{persona}_anchor_frontal_neutral.png"
        if anchor.exists():
            return anchor
        return None

    def _log_metrics(
        self,
        *,
        persona: str,
        gen_id: str,
        attempt: int,
        eval_result: EvaluationResult | None,
        decision: Decision,
        gen_ms: float,
        eval_ms: float,
        meta: ImageGenMeta,
    ) -> None:
        if self._metrics:
            self._metrics.log(
                persona=persona,
                generation_id=gen_id,
                attempt=attempt,
                eval_result=eval_result,
                decision_action=decision.action,
                generation_latency_ms=gen_ms,
                evaluation_latency_ms=eval_ms,
                prompt_hash=meta.prompt_hash,
                anchors_used=meta.anchors_used,
                model_name=meta.model,
            )

        if self._drift and eval_result is not None:
            self._drift.record(
                persona=persona,
                eval_result=eval_result,
                generation_id=gen_id,
                decision_action=decision.action,
            )
