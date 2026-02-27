"""Video identity evaluation: extract frames and evaluate each against an anchor."""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from personalab.identity.evaluator import IdentityEvaluator
from personalab.identity.frame_extractor import FrameExtractor
from personalab.identity.schemas import (
    EmbeddingTemporalStats,
    FrameEvaluationResult,
    GeometricTemporalStats,
    VideoEvaluationResult,
    CandidateResult,
    EmbeddingScore,
)

logger = logging.getLogger(__name__)


class StubVideoEvaluator:
    """Always returns perfect scores. Drop-in when no real evaluator is available."""

    def evaluate(
        self,
        anchor_path: str,
        video_data: bytes,
        *,
        max_frames: int = 8,
        strategy: str = "uniform",
    ) -> VideoEvaluationResult:
        return VideoEvaluationResult(
            frames_evaluated=0,
            mean_composite_score=1.0,
            min_composite_score=1.0,
            worst_frame_index=-1,
            frame_results=[],
            anchor_path=anchor_path,
        )


class VideoIdentityEvaluator:
    """Evaluate identity consistency in a video by scoring extracted frames.

    Composes a ``FrameExtractor`` (for pulling frames from video bytes)
    with an existing ``IdentityEvaluator`` (ArcFace + geometric scoring)
    and aggregates per-frame results.
    """

    def __init__(
        self,
        evaluator: IdentityEvaluator,
        frame_extractor: FrameExtractor,
    ) -> None:
        self._evaluator = evaluator
        self._frame_extractor = frame_extractor

    def evaluate(
        self,
        anchor_path: str,
        video_data: bytes,
        *,
        max_frames: int = 8,
        strategy: str = "uniform",
    ) -> VideoEvaluationResult:
        """Extract frames, evaluate each against the anchor, aggregate scores."""
        from personalab.identity.frame_extractor import ExtractedFrame

        extracted = self._frame_extractor.extract(
            video_data,
            strategy=strategy,
            max_frames=max_frames,
        )
        return self.evaluate_frames(anchor_path, extracted)

    def evaluate_frames(
        self,
        anchor_path: str,
        frames: list,
    ) -> VideoEvaluationResult:
        """Evaluate pre-extracted frames against the anchor.

        Accepts ``list[ExtractedFrame]`` so that the orchestrator can
        extract once and share frames with the quality evaluator.
        """
        if not frames:
            logger.warning("No frames to evaluate")
            return VideoEvaluationResult(anchor_path=anchor_path)

        frame_results: list[FrameEvaluationResult] = []
        min_score = float("inf")
        worst_idx = -1
        total_score = 0.0

        for ef in frames:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, ef.image)
                tmp_path = f.name

            try:
                eval_result = self._evaluator.evaluate(anchor_path, [tmp_path])
                if eval_result.candidates:
                    candidate = eval_result.candidates[0]
                else:
                    candidate = CandidateResult(
                        embedding=EmbeddingScore(
                            cosine_similarity=0.0, threshold=0.55, passed=False,
                        ),
                        geometric=None,
                        composite_score=0.0,
                        ok=False,
                        failure_reasons=["no_face_detected"],
                    )
            except Exception:
                logger.exception("Evaluation failed for frame %d", ef.frame_index)
                candidate = CandidateResult(
                    embedding=EmbeddingScore(
                        cosine_similarity=0.0, threshold=0.55, passed=False,
                    ),
                    geometric=None,
                    composite_score=0.0,
                    ok=False,
                    failure_reasons=["evaluation_error"],
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            score = candidate.composite_score
            total_score += score
            if score < min_score:
                min_score = score
                worst_idx = ef.frame_index

            frame_results.append(FrameEvaluationResult(
                frame_index=ef.frame_index,
                timestamp_s=ef.timestamp_s,
                eval_result=candidate,
            ))

        n = len(frame_results)
        mean_score = total_score / n if n > 0 else 0.0

        std_composite = _std([fr.eval_result.composite_score for fr in frame_results])
        emb_stats = _compute_embedding_temporal_stats(frame_results)
        geo_stats = _compute_geometric_temporal_stats(frame_results)

        return VideoEvaluationResult(
            frames_evaluated=n,
            mean_composite_score=mean_score,
            std_composite_score=std_composite,
            min_composite_score=min_score if min_score != float("inf") else 0.0,
            worst_frame_index=worst_idx,
            frame_results=frame_results,
            anchor_path=anchor_path,
            embedding_stats=emb_stats,
            geometric_stats=geo_stats,
        )


# ---------------------------------------------------------------------------
# Temporal statistics helpers
# ---------------------------------------------------------------------------

def _std(values: list[float]) -> float:
    """Population standard deviation (0.0 for fewer than 2 values)."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.std(arr, ddof=0))


def _compute_embedding_temporal_stats(
    frames: list[FrameEvaluationResult],
) -> EmbeddingTemporalStats | None:
    sims = [
        fr.eval_result.embedding.cosine_similarity
        for fr in frames
        if fr.eval_result.embedding is not None
    ]
    if not sims:
        return None
    arr = np.array(sims, dtype=np.float64)
    return EmbeddingTemporalStats(
        mean_cosine=round(float(np.mean(arr)), 6),
        std_cosine=round(float(np.std(arr, ddof=0)), 6),
        min_cosine=round(float(np.min(arr)), 6),
    )


def _compute_geometric_temporal_stats(
    frames: list[FrameEvaluationResult],
) -> GeometricTemporalStats | None:
    """Compute variance of each landmark ratio across frames.

    Each frame's ``landmark_deltas`` contains the absolute delta between
    anchor and candidate per ratio.  Here we treat those deltas as a
    time-series and compute their variance to detect *temporal drift*
    (ratios that fluctuate significantly across frames).
    """
    ratio_series: dict[str, list[float]] = {}
    for fr in frames:
        geo = fr.eval_result.geometric
        if geo is None:
            continue
        for name, delta in geo.landmark_deltas.items():
            ratio_series.setdefault(name, []).append(delta)

    if not ratio_series:
        return None

    variances: dict[str, float] = {}
    for name, deltas in ratio_series.items():
        arr = np.array(deltas, dtype=np.float64)
        variances[name] = round(float(np.var(arr, ddof=0)), 8)

    max_var = max(variances.values()) if variances else 0.0
    mean_var = float(np.mean(list(variances.values()))) if variances else 0.0

    return GeometricTemporalStats(
        ratio_variances=variances,
        max_ratio_variance=round(max_var, 8),
        mean_ratio_variance=round(mean_var, 8),
    )
