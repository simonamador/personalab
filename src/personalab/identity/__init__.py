"""Identity: anchor packs, evaluators, retry policy, drift tracking, video evaluation."""

from personalab.identity.anchor_pack import AnchorPack, load_anchor_pack
from personalab.identity.anchor_selector import select_anchor
from personalab.identity.identity_policy import IdentityPolicy, Decision
from personalab.identity.evaluator import IdentityEvaluator, StubEvaluator
from personalab.identity.composite_evaluator import CompositeEvaluator
from personalab.identity.drift import DriftTracker
from personalab.identity.factory import build_evaluator
from personalab.identity.frame_extractor import FrameExtractor, ExtractedFrame
from personalab.identity.video_evaluator import VideoIdentityEvaluator, StubVideoEvaluator
from personalab.identity.schemas import (
    EmbeddingScore,
    EmbeddingTemporalStats,
    GeometricTemporalStats,
    LandmarkScore,
    CandidateResult,
    EvaluationResult,
    FrameEvaluationResult,
    VideoEvaluationResult,
    DriftRecord,
    DriftSummary,
)

__all__ = [
    "AnchorPack",
    "load_anchor_pack",
    "select_anchor",
    "IdentityPolicy",
    "Decision",
    "IdentityEvaluator",
    "StubEvaluator",
    "CompositeEvaluator",
    "DriftTracker",
    "EmbeddingScore",
    "EmbeddingTemporalStats",
    "GeometricTemporalStats",
    "LandmarkScore",
    "CandidateResult",
    "EvaluationResult",
    "FrameEvaluationResult",
    "VideoEvaluationResult",
    "FrameExtractor",
    "ExtractedFrame",
    "VideoIdentityEvaluator",
    "StubVideoEvaluator",
    "DriftRecord",
    "DriftSummary",
    "build_evaluator",
]

# Lazy-loaded optional exports (require insightface)
def __getattr__(name: str):
    if name == "FaceRuntime":
        from personalab.identity.face_runtime import FaceRuntime
        return FaceRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
