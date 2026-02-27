"""Factory for building evaluators from config with graceful degradation."""

from __future__ import annotations

import logging
from typing import Any

from personalab.identity.evaluator import IdentityEvaluator, StubEvaluator

logger = logging.getLogger(__name__)


def build_evaluator(evaluation_cfg: dict[str, Any]) -> IdentityEvaluator:
    """Build the best available evaluator from the evaluation config section.

    Creates a single shared ``FaceRuntime`` and injects it into both
    ``ArcFaceEvaluator`` and ``GeometricEvaluator`` to avoid duplicate
    model loading.

    Falls back to StubEvaluator (with warning) if insightface is not installed.
    """
    try:
        from personalab.identity.face_runtime import FaceRuntime
        from personalab.identity.arcface_evaluator import ArcFaceEvaluator
        from personalab.identity.geometric_evaluator import GeometricEvaluator
        from personalab.identity.composite_evaluator import CompositeEvaluator
    except ImportError:
        logger.warning(
            "insightface not installed -- falling back to StubEvaluator. "
            "Install with: pip install personalab[eval]"
        )
        return StubEvaluator()

    emb_cfg = evaluation_cfg.get("embedding", {})
    geo_cfg = evaluation_cfg.get("geometric", {})
    scoring_cfg = evaluation_cfg.get("scoring", {})
    weights = scoring_cfg.get("weights", {})

    runtime = FaceRuntime(
        model_name=emb_cfg.get("model_name", "buffalo_l"),
    )

    arcface = ArcFaceEvaluator(
        runtime=runtime,
        similarity_threshold=emb_cfg.get("similarity_threshold", 0.55),
    )

    geometric = None
    if geo_cfg.get("enabled", True):
        geometric = GeometricEvaluator(
            runtime=runtime,
            max_normalized_error=geo_cfg.get("max_normalized_error", 0.08),
        )

    return CompositeEvaluator(
        embedding_evaluator=arcface,
        geometric_evaluator=geometric,
        weight_embedding=weights.get("embedding", 0.7),
        weight_geometric=weights.get("geometric", 0.3),
    )
