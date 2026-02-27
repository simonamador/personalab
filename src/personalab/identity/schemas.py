"""Typed schemas for identity evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingScore(BaseModel):
    """Result of embedding-based (ArcFace) identity comparison."""

    cosine_similarity: float = Field(..., description="Cosine similarity between anchor and candidate embeddings")
    threshold: float = Field(..., description="Minimum similarity to pass")
    passed: bool = Field(..., description="Whether cosine_similarity >= threshold")


class LandmarkScore(BaseModel):
    """Result of geometric landmark comparison."""

    normalized_error: float = Field(..., description="Mean normalized landmark distance (lower is better)")
    max_allowed: float = Field(..., description="Maximum normalized error to pass")
    passed: bool = Field(..., description="Whether normalized_error <= max_allowed")
    landmark_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-ratio deltas: inter_eye, nose_chin, jaw_width, etc.",
    )


class CandidateResult(BaseModel):
    """Evaluation result for a single candidate image."""

    embedding: EmbeddingScore | None = Field(default=None, description="Embedding-based score (None if not computed)")
    geometric: LandmarkScore | None = Field(default=None, description="Landmark-based score (None if not computed)")
    composite_score: float = Field(..., description="Weighted composite score (0.0 - 1.0)")
    ok: bool = Field(..., description="Whether the candidate passes all enabled checks")
    failure_reasons: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Full evaluation result for one or more candidates against an anchor."""

    candidates: list[CandidateResult] = Field(default_factory=list)
    anchor_path: str = Field(default="", description="Path to the anchor image used")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def first(self) -> CandidateResult:
        """Convenience: return the first candidate result (most common single-candidate case)."""
        if not self.candidates:
            raise IndexError("EvaluationResult has no candidates")
        return self.candidates[0]


class FrameEvaluationResult(BaseModel):
    """Identity evaluation result for a single video frame."""

    frame_index: int = Field(..., description="Frame index in the source video")
    timestamp_s: float = Field(..., description="Frame timestamp in seconds")
    eval_result: CandidateResult = Field(..., description="Identity evaluation for this frame")


class EmbeddingTemporalStats(BaseModel):
    """Distribution of cosine similarities across video frames (all vs anchor)."""

    mean_cosine: float = Field(0.0, description="Mean cosine similarity across frames")
    std_cosine: float = Field(0.0, description="Std deviation of cosine similarities")
    min_cosine: float = Field(0.0, description="Worst-frame cosine similarity")


class GeometricTemporalStats(BaseModel):
    """Temporal stability of normalised landmark ratios across video frames."""

    ratio_variances: dict[str, float] = Field(
        default_factory=dict,
        description="Variance of each landmark ratio across frames",
    )
    max_ratio_variance: float = Field(0.0, description="Largest per-ratio variance")
    mean_ratio_variance: float = Field(0.0, description="Mean of per-ratio variances")


class VideoEvaluationResult(BaseModel):
    """Aggregated identity evaluation across multiple video frames."""

    frames_evaluated: int = Field(0, description="Number of frames successfully evaluated")
    mean_composite_score: float = Field(0.0, description="Mean composite score across frames")
    std_composite_score: float = Field(0.0, description="Std deviation of composite scores")
    min_composite_score: float = Field(0.0, description="Minimum composite score (worst frame)")
    worst_frame_index: int = Field(-1, description="Frame index with the lowest score")
    frame_results: list[FrameEvaluationResult] = Field(default_factory=list)
    anchor_path: str = Field(default="", description="Path to the anchor image used")
    embedding_stats: EmbeddingTemporalStats | None = Field(
        default=None, description="Temporal distribution of embedding cosine similarities",
    )
    geometric_stats: GeometricTemporalStats | None = Field(
        default=None, description="Temporal stability of geometric landmark ratios",
    )

    def as_evaluation_result(self) -> EvaluationResult:
        """Convert to a standard EvaluationResult using the mean scores.

        This allows reuse of IdentityPolicy for accept/retry/reject decisions.
        """
        if not self.frame_results:
            return EvaluationResult(anchor_path=self.anchor_path, candidates=[])

        mean_embedding_sim = 0.0
        mean_geo_error = 0.0
        embedding_count = 0
        geo_count = 0

        for fr in self.frame_results:
            cr = fr.eval_result
            if cr.embedding is not None:
                mean_embedding_sim += cr.embedding.cosine_similarity
                embedding_count += 1
            if cr.geometric is not None:
                mean_geo_error += cr.geometric.normalized_error
                geo_count += 1

        if embedding_count > 0:
            mean_embedding_sim /= embedding_count
        if geo_count > 0:
            mean_geo_error /= geo_count

        ref = self.frame_results[0].eval_result
        embedding_score = None
        if ref.embedding is not None:
            embedding_score = EmbeddingScore(
                cosine_similarity=mean_embedding_sim,
                threshold=ref.embedding.threshold,
                passed=mean_embedding_sim >= ref.embedding.threshold,
            )

        geo_score = None
        if ref.geometric is not None:
            geo_score = LandmarkScore(
                normalized_error=mean_geo_error,
                max_allowed=ref.geometric.max_allowed,
                passed=mean_geo_error <= ref.geometric.max_allowed,
                landmark_deltas={},
            )

        candidate = CandidateResult(
            embedding=embedding_score,
            geometric=geo_score,
            composite_score=self.mean_composite_score,
            ok=self.mean_composite_score >= (ref.embedding.threshold if ref.embedding else 0.55),
            failure_reasons=[],
        )
        return EvaluationResult(
            candidates=[candidate],
            anchor_path=self.anchor_path,
        )


class DriftRecord(BaseModel):
    """Single entry in a persona's drift timeline."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    generation_id: str = ""
    persona: str = ""
    composite_score: float = 0.0
    embedding_similarity: float | None = None
    geometric_error: float | None = None
    decision_action: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DriftSummary(BaseModel):
    """Rolling summary of drift for a persona."""

    persona: str = ""
    window_size: int = 0
    count: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    trend: str = Field(default="stable", description="improving | degrading | stable")
