"""Typed schemas for visual quality evaluation (separate from identity)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrameSharpness(BaseModel):
    """Sharpness measurement for a single video frame."""

    frame_index: int = Field(..., description="Frame index in the source video")
    laplacian_variance: float = Field(..., description="Variance of Laplacian (higher = sharper)")
    anchor_ratio: float = Field(
        ..., description="frame laplacian_variance / anchor laplacian_variance",
    )


class IlluminationCheck(BaseModel):
    """Histogram distance between two key frames."""

    frame_index_a: int = Field(..., description="First frame index")
    frame_index_b: int = Field(..., description="Second frame index")
    histogram_distance: float = Field(..., description="Distance between normalised histograms")
    method: str = Field("bhattacharyya", description="Comparison method used")


class VideoQualityResult(BaseModel):
    """Aggregated visual quality assessment across video frames."""

    mean_sharpness_ratio: float = Field(0.0, description="Mean frame/anchor Laplacian ratio")
    min_sharpness_ratio: float = Field(0.0, description="Worst-frame sharpness ratio")
    std_sharpness_ratio: float = Field(0.0, description="Std deviation of sharpness ratios")
    frame_sharpness: list[FrameSharpness] = Field(default_factory=list)
    sharpness_ok: bool = Field(True, description="All frames above min_ratio threshold")

    illumination_checks: list[IlluminationCheck] = Field(default_factory=list)
    mean_histogram_distance: float = Field(0.0, description="Mean histogram distance across pairs")
    max_histogram_distance: float = Field(0.0, description="Worst histogram distance")
    illumination_ok: bool = Field(True, description="All pairs below max distance threshold")


class PostProcessingAdvice(BaseModel):
    """Advisory output from the post-processing gate."""

    identity_stable: bool = Field(..., description="Whether identity verification passed")
    quality_degraded: bool = Field(False, description="Whether visual quality is below acceptable")
    degradation_type: str = Field(
        "none",
        description="none | textural | illumination | both",
    )
    recommend_sharpening: bool = Field(False)
    recommend_super_resolution: bool = Field(False)
    notes: str = Field("")
