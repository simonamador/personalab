"""Runner abstraction: SyncRunner (baseline) and AsyncRunner (parallel).

``Runner`` is the top-level entry-point for executing a batch of generation
jobs.  Two implementations guarantee that the same ``IdentityPolicy`` and
scoring logic govern decisions regardless of execution strategy:

* **SyncRunner** -- sequential, fully deterministic baseline.  Delegates
  directly to the existing ``GenerationOrchestrator`` / ``VideoOrchestrator``.
* **AsyncRunner** -- runs jobs concurrently via ``asyncio``, offloads
  evaluation to a ``ProcessPoolExecutor``, and enforces per-provider
  semaphores and rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from personalab.generation.image_generator import ImageGenerator
from personalab.generation.video_generator import VideoGenerator
from personalab.identity.evaluator import IdentityEvaluator
from personalab.identity.identity_policy import Decision, IdentityPolicy
from personalab.identity.drift import DriftTracker
from personalab.identity.schemas import EvaluationResult
from personalab.llm.client import (
    LLMClient, TextPart, BytesPart, ContentPart,
    LLMImageResponse, LLMVideoResponse,
)
from personalab.observability.metrics_logger import GenerationMetricsLogger
from personalab.schemas.prompts import ImagePrompt, VideoPrompt, ImageGenMeta, VideoGenMeta

from personalab.runtime.job import JobSpec, JobResult
from personalab.runtime.concurrency import ProviderSemaphore, RateLimiter
from personalab.runtime.eval_pool import EvaluationPool
from personalab.runtime.retry_policy import RetryPolicy

logger = logging.getLogger(__name__)


# ---- abstract base --------------------------------------------------------

class Runner(ABC):
    """Execute a batch of generation jobs and return results."""

    @abstractmethod
    def run(self, jobs: Sequence[JobSpec]) -> list[JobResult]:
        """Run all *jobs* and return one result per job (same order)."""
        ...


# ---- SyncRunner -----------------------------------------------------------

class SyncRunner(Runner):
    """Sequential runner that delegates to the existing orchestrators.

    Produces identical results to calling ``GenerationOrchestrator.run()``
    or ``VideoOrchestrator.run()`` directly -- the only addition is
    ``JobSpec`` / ``JobResult`` wrapping and retry-policy logging.
    """

    def __init__(
        self,
        *,
        image_generator: ImageGenerator,
        video_generator: VideoGenerator | None = None,
        evaluator: IdentityEvaluator,
        policy: IdentityPolicy,
        retry_policy: RetryPolicy | None = None,
        drift_tracker: DriftTracker | None = None,
        metrics_logger: GenerationMetricsLogger | None = None,
        output_dir: str | Path = "./generated_content",
        references_root: str | Path = "./references",
    ) -> None:
        from personalab.orchestration.generation_orchestrator import GenerationOrchestrator
        self._img_orch = GenerationOrchestrator(
            image_generator=image_generator,
            evaluator=evaluator,
            policy=policy,
            drift_tracker=drift_tracker,
            metrics_logger=metrics_logger,
            output_dir=output_dir,
            references_root=references_root,
        )
        self._vid_gen = video_generator
        self._evaluator = evaluator
        self._policy = policy
        self._retry = retry_policy or RetryPolicy(max_retries=policy.max_attempts)
        self._drift = drift_tracker
        self._metrics = metrics_logger
        self._output_dir = Path(output_dir)
        self._references_root = Path(references_root)

    def run(self, jobs: Sequence[JobSpec]) -> list[JobResult]:
        results: list[JobResult] = []
        for job in jobs:
            started = datetime.now(timezone.utc).isoformat()
            try:
                if job.asset_type == "image":
                    jr = self._run_image(job, started)
                else:
                    jr = self._run_video(job, started)
            except Exception as exc:
                logger.exception("Job %s failed", job.job_id)
                jr = JobResult(
                    job_id=job.job_id,
                    seed=job.seed,
                    config_hash=job.config_hash,
                    asset_type=job.asset_type,
                    started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )
            results.append(jr)
        return results

    def _run_image(self, job: JobSpec, started: str) -> JobResult:
        assert isinstance(job.prompt, ImagePrompt)
        orch_result = self._img_orch.run(
            persona=job.persona,
            scenario_template=job.scenario_template,
            image_prompt=job.prompt,
            generation_id=job.generation_id or job.job_id,
            output_prefix=job.output_prefix,
        )
        return JobResult(
            job_id=job.job_id,
            seed=job.seed,
            config_hash=job.config_hash,
            asset_type="image",
            accepted=orch_result.accepted,
            final_attempt=orch_result.final_attempt,
            final_score=orch_result.final_score,
            final_path=orch_result.final_image_path,
            started_at=started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            attempts_summary=[a.model_dump() for a in orch_result.attempts],
        )

    def _run_video(self, job: JobSpec, started: str) -> JobResult:
        if self._vid_gen is None:
            raise RuntimeError("VideoGenerator not provided to SyncRunner")
        assert isinstance(job.prompt, VideoPrompt)

        from personalab.orchestration.video_orchestrator import VideoOrchestrator
        from personalab.identity.video_evaluator import VideoIdentityEvaluator, StubVideoEvaluator
        from personalab.identity.frame_extractor import FrameExtractor

        vid_evaluator: Any
        try:
            vid_evaluator = VideoIdentityEvaluator(
                evaluator=self._evaluator,
                frame_extractor=FrameExtractor(),
            )
        except Exception:
            vid_evaluator = StubVideoEvaluator()

        vid_orch = VideoOrchestrator(
            video_generator=self._vid_gen,
            video_evaluator=vid_evaluator,
            policy=self._policy,
            drift_tracker=self._drift,
            metrics_logger=self._metrics,
            output_dir=self._output_dir,
            references_root=self._references_root,
        )
        orch_result = vid_orch.run(
            persona=job.persona,
            scenario_template=job.scenario_template,
            video_prompt=job.prompt,
            generation_id=job.generation_id or job.job_id,
            output_prefix=job.output_prefix,
        )
        return JobResult(
            job_id=job.job_id,
            seed=job.seed,
            config_hash=job.config_hash,
            asset_type="video",
            accepted=orch_result.accepted,
            final_attempt=orch_result.final_attempt,
            final_score=orch_result.final_score,
            final_path=orch_result.final_video_path,
            started_at=started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            attempts_summary=[a.model_dump() for a in orch_result.attempts],
        )


# ---- AsyncRunner ----------------------------------------------------------

class AsyncRunner(Runner):
    """Concurrent runner: asyncio for I/O, ProcessPoolExecutor for eval.

    Semaphores and rate-limits prevent provider throttling.  The same
    ``IdentityPolicy.decide()`` governs accept/retry/reject -- parallelism
    only changes wall-clock time, never metrics or decisions.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        image_generator: ImageGenerator,
        video_generator: VideoGenerator | None = None,
        policy: IdentityPolicy,
        retry_policy: RetryPolicy | None = None,
        eval_pool: EvaluationPool | None = None,
        provider_semaphore: ProviderSemaphore | None = None,
        rate_limiter: RateLimiter | None = None,
        drift_tracker: DriftTracker | None = None,
        metrics_logger: GenerationMetricsLogger | None = None,
        output_dir: str | Path = "./generated_content",
        references_root: str | Path = "./references",
        provider_key: str = "gemini",
    ) -> None:
        self._client = client
        self._img_gen = image_generator
        self._vid_gen = video_generator
        self._policy = policy
        self._retry = retry_policy or RetryPolicy(max_retries=policy.max_attempts)
        self._eval_pool = eval_pool
        self._sem = provider_semaphore or ProviderSemaphore()
        self._rl = rate_limiter or RateLimiter()
        self._drift = drift_tracker
        self._metrics = metrics_logger
        self._output_dir = Path(output_dir)
        self._references_root = Path(references_root)
        self._provider_key = provider_key

    # -- public API ---------------------------------------------------------

    def run(self, jobs: Sequence[JobSpec]) -> list[JobResult]:
        """Synchronous entry-point; spins up an event loop internally."""
        return asyncio.run(self._run_all(list(jobs)))

    async def run_async(self, jobs: Sequence[JobSpec]) -> list[JobResult]:
        """Async entry-point for callers already inside an event loop."""
        return await self._run_all(list(jobs))

    # -- internal -----------------------------------------------------------

    async def _run_all(self, jobs: list[JobSpec]) -> list[JobResult]:
        tasks = [self._run_job(job) for job in jobs]
        return await asyncio.gather(*tasks)

    async def _run_job(self, job: JobSpec) -> JobResult:
        started = datetime.now(timezone.utc).isoformat()
        try:
            if job.asset_type == "image":
                return await self._run_image_job(job, started)
            else:
                return await self._run_video_job(job, started)
        except Exception as exc:
            logger.exception("Async job %s failed", job.job_id)
            return JobResult(
                job_id=job.job_id,
                seed=job.seed,
                config_hash=job.config_hash,
                asset_type=job.asset_type,
                started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )

    async def _run_image_job(self, job: JobSpec, started: str) -> JobResult:
        assert isinstance(job.prompt, ImagePrompt)
        gen_id = job.generation_id or job.job_id
        prompt = job.prompt.model_copy()
        anchor_path = self._resolve_anchor(job.persona)

        attempts_summary: list[dict[str, Any]] = []
        accepted = False
        final_attempt = 0
        final_score = 0.0
        final_path: str | None = None

        attempt = 0
        while attempt < self._policy.max_attempts:
            attempt += 1

            async with self._sem[self._provider_key]:
                await self._rl.acquire(self._provider_key)
                t0 = time.perf_counter()
                img_bytes, meta, prompt_used = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._img_gen.generate_asset_image(
                        persona_name=job.persona,
                        scenario_template=job.scenario_template,
                        image_prompt=prompt,
                    ),
                )
                gen_ms = (time.perf_counter() - t0) * 1000

            RetryPolicy.log_attempt(
                job_id=job.job_id, seed=job.seed, attempt=attempt,
                prompt_hash=meta.prompt_hash if isinstance(meta, ImageGenMeta) else "",
                prompt_applied=prompt_used if isinstance(prompt_used, dict) else {},
                decision=Decision(action="PENDING", score=0.0),
            )

            if not img_bytes:
                attempts_summary.append({"attempt": attempt, "generation_ms": gen_ms, "error": "no_bytes"})
                continue

            out_path = self._output_dir / job.persona / f"{job.output_prefix}_{attempt}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)

            eval_result: EvaluationResult | None = None
            eval_ms = 0.0
            if anchor_path and self._eval_pool:
                t1 = time.perf_counter()
                eval_result = await self._eval_pool.evaluate(str(anchor_path), [str(out_path)])
                eval_ms = (time.perf_counter() - t1) * 1000

            if eval_result is not None:
                decision = self._policy.decide(attempt, eval_result)
            else:
                decision = Decision(action="ACCEPT", notes="no_anchor_or_pool", score=1.0)

            RetryPolicy.log_attempt(
                job_id=job.job_id, seed=job.seed, attempt=attempt,
                prompt_hash=meta.prompt_hash if isinstance(meta, ImageGenMeta) else "",
                prompt_applied=prompt_used if isinstance(prompt_used, dict) else {},
                decision=decision,
            )

            if self._metrics:
                self._metrics.log(
                    persona=job.persona, generation_id=gen_id, attempt=attempt,
                    eval_result=eval_result, decision_action=decision.action,
                    generation_latency_ms=gen_ms, evaluation_latency_ms=eval_ms,
                    prompt_hash=meta.prompt_hash if isinstance(meta, ImageGenMeta) else "",
                    anchors_used=meta.anchors_used if isinstance(meta, ImageGenMeta) else [],
                    model_name=meta.model if isinstance(meta, ImageGenMeta) else "",
                )

            attempts_summary.append({
                "attempt": attempt, "generation_ms": gen_ms, "evaluation_ms": eval_ms,
                "decision": decision.action, "score": decision.score,
                "image_path": str(out_path),
            })

            if decision.action == "ACCEPT":
                accepted = True
                final_attempt = attempt
                final_score = decision.score
                final_path = str(out_path)
                break
            elif decision.action == "RETRY" and self._retry.should_retry(attempt, decision):
                await self._retry.wait_async(attempt)
                prompt.policy_overrides = decision.patches.get("policy_overrides")
            else:
                final_attempt = attempt
                final_score = decision.score
                break

        if not accepted and final_attempt == 0:
            final_attempt = attempt

        return JobResult(
            job_id=job.job_id, seed=job.seed, config_hash=job.config_hash,
            asset_type="image", accepted=accepted,
            final_attempt=final_attempt, final_score=final_score, final_path=final_path,
            started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
            attempts_summary=attempts_summary,
        )

    async def _run_video_job(self, job: JobSpec, started: str) -> JobResult:
        if self._vid_gen is None:
            raise RuntimeError("VideoGenerator not provided to AsyncRunner")
        assert isinstance(job.prompt, VideoPrompt)
        gen_id = job.generation_id or job.job_id
        prompt = job.prompt.model_copy()
        anchor_path = self._resolve_anchor(job.persona)

        attempts_summary: list[dict[str, Any]] = []
        accepted = False
        final_attempt = 0
        final_score = 0.0
        final_path: str | None = None

        attempt = 0
        while attempt < self._policy.max_attempts:
            attempt += 1

            async with self._sem[self._provider_key]:
                await self._rl.acquire(self._provider_key)
                t0 = time.perf_counter()
                resp, meta = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._vid_gen.generate_scenario_video(
                        persona_name=job.persona,
                        video_scenario_template=job.scenario_template,
                        video_prompt=prompt,
                    ),
                )
                gen_ms = (time.perf_counter() - t0) * 1000

            video_bytes = resp.video_data
            if not video_bytes:
                attempts_summary.append({"attempt": attempt, "generation_ms": gen_ms, "error": "no_bytes"})
                continue

            out_path = self._output_dir / job.persona / f"{job.output_prefix}_{attempt}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(video_bytes)

            eval_result: EvaluationResult | None = None
            eval_ms = 0.0
            if anchor_path and self._eval_pool:
                t1 = time.perf_counter()
                eval_result = await self._eval_pool.evaluate(str(anchor_path), [str(out_path)])
                eval_ms = (time.perf_counter() - t1) * 1000

            if eval_result is not None:
                decision = self._policy.decide(attempt, eval_result)
            else:
                decision = Decision(action="ACCEPT", notes="no_anchor_or_pool", score=1.0)

            if self._metrics:
                self._metrics.log(
                    persona=job.persona, generation_id=gen_id, attempt=attempt,
                    eval_result=eval_result, decision_action=decision.action,
                    generation_latency_ms=gen_ms, evaluation_latency_ms=eval_ms,
                    prompt_hash=meta.prompt_hash if isinstance(meta, VideoGenMeta) else "",
                    anchors_used=meta.anchors_used if isinstance(meta, VideoGenMeta) else [],
                    model_name=meta.model if isinstance(meta, VideoGenMeta) else "",
                )

            attempts_summary.append({
                "attempt": attempt, "generation_ms": gen_ms, "evaluation_ms": eval_ms,
                "decision": decision.action, "score": decision.score,
                "video_path": str(out_path),
            })

            if decision.action == "ACCEPT":
                accepted = True
                final_attempt = attempt
                final_score = decision.score
                final_path = str(out_path)
                break
            elif decision.action == "RETRY" and self._retry.should_retry(attempt, decision):
                await self._retry.wait_async(attempt)
                prompt.policy_overrides = decision.patches.get("policy_overrides")
            else:
                final_attempt = attempt
                final_score = decision.score
                break

        if not accepted and final_attempt == 0:
            final_attempt = attempt

        return JobResult(
            job_id=job.job_id, seed=job.seed, config_hash=job.config_hash,
            asset_type="video", accepted=accepted,
            final_attempt=final_attempt, final_score=final_score, final_path=final_path,
            started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
            attempts_summary=attempts_summary,
        )

    def _resolve_anchor(self, persona: str) -> Path | None:
        anchor = self._references_root / persona / "anchors" / f"{persona}_anchor_frontal_neutral.png"
        if anchor.exists():
            return anchor
        return None
