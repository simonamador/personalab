"""Tests for the PostProcessingGate."""

import pytest

from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EmbeddingTemporalStats,
    FrameEvaluationResult,
    GeometricTemporalStats,
    LandmarkScore,
    VideoEvaluationResult,
)
from personalab.quality.post_processing_gate import PostProcessingGate
from personalab.quality.schemas import (
    FrameSharpness,
    IlluminationCheck,
    PostProcessingAdvice,
    VideoQualityResult,
)


def _identity_result(
    *,
    mean_composite: float = 0.7,
    std_composite: float = 0.02,
    emb_mean: float = 0.72,
    emb_std: float = 0.03,
    emb_min: float = 0.65,
    geo_mean_var: float = 0.0005,
    frames: int = 4,
) -> VideoEvaluationResult:
    fr = []
    for i in range(frames):
        fr.append(FrameEvaluationResult(
            frame_index=i,
            timestamp_s=i * 0.5,
            eval_result=CandidateResult(
                embedding=EmbeddingScore(cosine_similarity=emb_mean, threshold=0.55, passed=True),
                geometric=LandmarkScore(
                    normalized_error=0.04, max_allowed=0.08, passed=True, landmark_deltas={},
                ),
                composite_score=mean_composite,
                ok=True,
            ),
        ))
    return VideoEvaluationResult(
        frames_evaluated=frames,
        mean_composite_score=mean_composite,
        std_composite_score=std_composite,
        min_composite_score=mean_composite - 0.05,
        worst_frame_index=0,
        frame_results=fr,
        embedding_stats=EmbeddingTemporalStats(
            mean_cosine=emb_mean, std_cosine=emb_std, min_cosine=emb_min,
        ),
        geometric_stats=GeometricTemporalStats(
            ratio_variances={"inter_eye": geo_mean_var},
            max_ratio_variance=geo_mean_var,
            mean_ratio_variance=geo_mean_var,
        ),
    )


def _quality_result(
    *, sharpness_ok: bool = True, illumination_ok: bool = True,
) -> VideoQualityResult:
    return VideoQualityResult(
        mean_sharpness_ratio=0.8 if sharpness_ok else 0.2,
        min_sharpness_ratio=0.6 if sharpness_ok else 0.1,
        std_sharpness_ratio=0.05,
        sharpness_ok=sharpness_ok,
        illumination_ok=illumination_ok,
        mean_histogram_distance=0.1 if illumination_ok else 0.5,
        max_histogram_distance=0.15 if illumination_ok else 0.6,
    )


class TestPostProcessingGate:
    def test_all_ok_no_recommendations(self):
        gate = PostProcessingGate()
        advice = gate.evaluate(
            _identity_result(),
            _quality_result(sharpness_ok=True, illumination_ok=True),
        )
        assert isinstance(advice, PostProcessingAdvice)
        assert advice.identity_stable is True
        assert advice.quality_degraded is False
        assert advice.recommend_sharpening is False
        assert advice.recommend_super_resolution is False
        assert advice.degradation_type == "none"

    def test_textural_degradation_recommends_sharpening(self):
        gate = PostProcessingGate()
        advice = gate.evaluate(
            _identity_result(),
            _quality_result(sharpness_ok=False, illumination_ok=True),
        )
        assert advice.identity_stable is True
        assert advice.quality_degraded is True
        assert advice.degradation_type == "textural"
        assert advice.recommend_sharpening is True
        assert advice.recommend_super_resolution is True

    def test_identity_unstable_blocks_postprocessing(self):
        gate = PostProcessingGate()
        advice = gate.evaluate(
            _identity_result(mean_composite=0.3),
            _quality_result(sharpness_ok=False),
        )
        assert advice.identity_stable is False
        assert advice.recommend_sharpening is False
        assert advice.recommend_super_resolution is False
        assert "no_postprocessing" in advice.notes

    def test_high_embedding_std_blocks_postprocessing(self):
        gate = PostProcessingGate(max_embedding_std=0.04)
        advice = gate.evaluate(
            _identity_result(emb_std=0.08),
            _quality_result(sharpness_ok=False),
        )
        assert advice.identity_stable is False
        assert advice.recommend_sharpening is False

    def test_high_geometric_variance_blocks_postprocessing(self):
        gate = PostProcessingGate(max_geometric_variance=0.001)
        advice = gate.evaluate(
            _identity_result(geo_mean_var=0.005),
            _quality_result(sharpness_ok=False),
        )
        assert advice.identity_stable is False
        assert advice.recommend_sharpening is False

    def test_illumination_only_degradation(self):
        gate = PostProcessingGate()
        advice = gate.evaluate(
            _identity_result(),
            _quality_result(sharpness_ok=True, illumination_ok=False),
        )
        assert advice.identity_stable is True
        assert advice.quality_degraded is True
        assert advice.degradation_type == "illumination"
        assert advice.recommend_sharpening is False
        assert "no_auto_fix" in advice.notes

    def test_both_degradation(self):
        gate = PostProcessingGate()
        advice = gate.evaluate(
            _identity_result(),
            _quality_result(sharpness_ok=False, illumination_ok=False),
        )
        assert advice.degradation_type == "both"
        assert advice.recommend_sharpening is False
        assert "review_manually" in advice.notes

    def test_zero_frames_is_unstable(self):
        gate = PostProcessingGate()
        identity = VideoEvaluationResult(frames_evaluated=0, anchor_path="a.png")
        advice = gate.evaluate(identity, _quality_result())
        assert advice.identity_stable is False
