"""Video generation-evaluate-retry orchestrator.

Mirrors ``GenerationOrchestrator`` for images, but operates on video:
generate video -> extract frames -> evaluate identity + quality -> policy decision -> retry.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from personalab.generation.video_generator import VideoGenerator
from personalab.identity.identity_policy import Decision, IdentityPolicy
from personalab.identity.video_evaluator import VideoIdentityEvaluator
from personalab.identity.frame_extractor import FrameExtractor
from personalab.identity.schemas import VideoEvaluationResult
from personalab.identity.drift import DriftTracker
from personalab.quality.schemas import PostProcessingAdvice, VideoQualityResult
from personalab.quality.video_quality_evaluator import VideoQualityEvaluator
from personalab.quality.post_processing_gate import PostProcessingGate
from personalab.observability.metrics_logger import GenerationMetricsLogger
from personalab.schemas import VideoGenMeta, VideoPrompt

logger = logging.getLogger(__name__)


class VideoGenerationAttempt(BaseModel):
    """Record of a single video generation attempt."""

    attempt: int
    video_path: str | None = None
    video_bytes_size: int = 0
    eval_result: VideoEvaluationResult | None = None
    quality_result: VideoQualityResult | None = None
    post_processing_advice: PostProcessingAdvice | None = None
    decision: Decision | None = None
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    quality_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class VideoOrchestrationResult(BaseModel):
    """Final result of the video orchestration loop."""

    persona: str
    generation_id: str
    accepted: bool = False
    final_attempt: int = 0
    final_video_path: str | None = None
    final_score: float = 0.0
    attempts: list[VideoGenerationAttempt] = Field(default_factory=list)


class VideoOrchestrator:
    """Generate video -> extract frames -> evaluate identity + quality -> policy -> retry loop.

    All dependencies are injected; no global state.
    Quality evaluation and post-processing gate are optional.
    Policy decisions are always based on identity only; quality is advisory.
    """

    def __init__(
        self,
        *,
        video_generator: VideoGenerator,
        video_evaluator: VideoIdentityEvaluator,
        policy: IdentityPolicy,
        frame_extractor: FrameExtractor | None = None,
        quality_evaluator: VideoQualityEvaluator | None = None,
        post_processing_gate: PostProcessingGate | None = None,
        drift_tracker: DriftTracker | None = None,
        metrics_logger: GenerationMetricsLogger | None = None,
        output_dir: str | Path = "./generated_content",
        references_root: str | Path = "./references",
        max_frames: int = 8,
        frame_strategy: str = "uniform",
    ) -> None:
        self._vid_gen = video_generator
        self._evaluator = video_evaluator
        self._policy = policy
        self._frame_extractor = frame_extractor
        self._quality = quality_evaluator
        self._gate = post_processing_gate
        self._drift = drift_tracker
        self._metrics = metrics_logger
        self._output_dir = Path(output_dir)
        self._references_root = Path(references_root)
        self._max_frames = max_frames
        self._frame_strategy = frame_strategy

    def run(
        self,
        *,
        persona: str,
        scenario_template: dict[str, Any],
        video_prompt: VideoPrompt,
        generation_id: str | None = None,
        output_prefix: str = "vid",
    ) -> VideoOrchestrationResult:
        """Execute the full video generation-evaluation loop."""
        gen_id = generation_id or uuid.uuid4().hex[:12]
        anchor_path = self._resolve_anchor(persona)

        result = VideoOrchestrationResult(persona=persona, generation_id=gen_id)
        attempt = 0

        while attempt < self._policy.max_attempts:
            attempt += 1

            t0 = time.perf_counter()
            resp, meta = self._vid_gen.generate_scenario_video(
                persona_name=persona,
                video_scenario_template=scenario_template,
                video_prompt=video_prompt,
            )
            gen_ms = (time.perf_counter() - t0) * 1000

            rec = VideoGenerationAttempt(attempt=attempt, generation_ms=gen_ms)
            rec.meta = meta.model_dump() if isinstance(meta, VideoGenMeta) else {}

            video_bytes = resp.video_data
            if not video_bytes:
                logger.warning("Attempt %d: video generation returned no bytes", attempt)
                result.attempts.append(rec)
                continue

            out_path = self._output_dir / persona / f"{output_prefix}_{attempt}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(video_bytes)
            rec.video_path = str(out_path)
            rec.video_bytes_size = len(video_bytes)

            if anchor_path:
                extracted_frames = self._extract_frames(video_bytes)

                t1 = time.perf_counter()
                video_eval = self._evaluator.evaluate_frames(
                    str(anchor_path), extracted_frames,
                )
                eval_ms = (time.perf_counter() - t1) * 1000
                rec.eval_result = video_eval
                rec.evaluation_ms = eval_ms

                if self._quality and extracted_frames:
                    t2 = time.perf_counter()
                    quality_result = self._quality.evaluate(
                        str(anchor_path), extracted_frames,
                    )
                    quality_ms = (time.perf_counter() - t2) * 1000
                    rec.quality_result = quality_result
                    rec.quality_ms = quality_ms

                    if self._gate:
                        rec.post_processing_advice = self._gate.evaluate(
                            video_eval, quality_result,
                        )

                eval_result = video_eval.as_evaluation_result()
                decision = self._policy.decide(attempt, eval_result)
            else:
                video_eval = None
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
                result.final_video_path = str(out_path)
                result.final_score = decision.score
                logger.info("Video accepted at attempt %d (score=%.4f)", attempt, decision.score)
                break
            elif decision.action == "RETRY":
                logger.info(
                    "Video retry (attempt %d, score=%.4f): %s",
                    attempt, decision.score, decision.notes,
                )
                video_prompt.policy_overrides = decision.patches.get("policy_overrides")
            else:
                result.final_attempt = attempt
                result.final_score = decision.score
                logger.info("Video rejected final (attempt %d, score=%.4f)", attempt, decision.score)
                break

        if not result.accepted and result.final_attempt == 0:
            result.final_attempt = attempt

        return result

    def _extract_frames(self, video_bytes: bytes) -> list:
        """Extract frames using the shared FrameExtractor or the evaluator's own."""
        if self._frame_extractor is not None:
            return self._frame_extractor.extract(
                video_bytes,
                strategy=self._frame_strategy,
                max_frames=self._max_frames,
            )
        return self._evaluator._frame_extractor.extract(
            video_bytes,
            strategy=self._frame_strategy,
            max_frames=self._max_frames,
        )

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
        eval_result: Any,
        decision: Decision,
        gen_ms: float,
        eval_ms: float,
        meta: VideoGenMeta,
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
