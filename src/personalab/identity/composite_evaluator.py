"""Composite evaluator: weighted combination of embedding and geometric scores."""

from __future__ import annotations

import logging

from personalab.identity.evaluator import IdentityEvaluator
from personalab.identity.schemas import (
    CandidateResult,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class CompositeEvaluator:
    """Combines an embedding evaluator and an optional geometric evaluator with configurable weights.

    composite_score = w_embedding * cosine_similarity + w_geometric * (1 - normalized_error)

    When geometric evaluator is None, embedding weight is effectively 1.0.
    """

    def __init__(
        self,
        *,
        embedding_evaluator: IdentityEvaluator,
        geometric_evaluator: IdentityEvaluator | None = None,
        weight_embedding: float = 0.7,
        weight_geometric: float = 0.3,
    ) -> None:
        self._embedding = embedding_evaluator
        self._geometric = geometric_evaluator
        self._w_emb = weight_embedding
        self._w_geo = weight_geometric

        if geometric_evaluator is None:
            self._w_emb = 1.0
            self._w_geo = 0.0

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        emb_result = self._embedding.evaluate(anchor_path, candidate_paths)

        geo_result = None
        if self._geometric is not None:
            geo_result = self._geometric.evaluate(anchor_path, candidate_paths)

        candidates: list[CandidateResult] = []
        for i, cpath in enumerate(candidate_paths):
            emb_cand = emb_result.candidates[i] if i < len(emb_result.candidates) else None
            geo_cand = geo_result.candidates[i] if geo_result and i < len(geo_result.candidates) else None

            emb_score = emb_cand.embedding if emb_cand else None
            geo_score = geo_cand.geometric if geo_cand else None

            emb_sim = emb_score.cosine_similarity if emb_score else 0.0
            geo_val = (1.0 - geo_score.normalized_error) if geo_score else 0.0

            composite = self._w_emb * emb_sim + self._w_geo * geo_val
            composite = round(max(0.0, min(1.0, composite)), 6)

            failures: list[str] = []
            if emb_score and not emb_score.passed:
                failures.append("embedding_below_threshold")
            if geo_score and not geo_score.passed:
                failures.append("geometric_error_above_threshold")
            if emb_cand and not emb_cand.ok and emb_cand.failure_reasons:
                for r in emb_cand.failure_reasons:
                    if r not in failures:
                        failures.append(r)

            ok = (emb_score.passed if emb_score else True) and (geo_score.passed if geo_score else True)

            candidates.append(
                CandidateResult(
                    embedding=emb_score,
                    geometric=geo_score,
                    composite_score=composite,
                    ok=ok,
                    failure_reasons=failures,
                )
            )

        return EvaluationResult(candidates=candidates, anchor_path=anchor_path)
