"""ArcFace embedding evaluator using a shared FaceRuntime.

Requires the optional [eval] dependency group: pip install personalab[eval]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from personalab.identity.face_runtime import FaceRuntime
from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return max(0.0, dot / norm)


class ArcFaceEvaluator:
    """Identity evaluator based on ArcFace 512-d embeddings and cosine similarity.

    Receives a shared ``FaceRuntime`` via dependency injection.
    Caches the anchor embedding across calls so repeated retries don't
    re-extract from the same anchor image.
    """

    def __init__(
        self,
        *,
        runtime: FaceRuntime,
        similarity_threshold: float = 0.55,
    ) -> None:
        self._runtime = runtime
        self.similarity_threshold = similarity_threshold
        self._anchor_cache: dict[str, np.ndarray] = {}

    def get_anchor_embedding(self, anchor_path: str) -> np.ndarray | None:
        """Return cached anchor embedding, extracting on first call."""
        if anchor_path not in self._anchor_cache:
            emb = self._runtime.extract_embedding(anchor_path)
            if emb is not None:
                self._anchor_cache[anchor_path] = emb
        return self._anchor_cache.get(anchor_path)

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        anchor_emb = self.get_anchor_embedding(anchor_path)
        if anchor_emb is None:
            return _no_face_result(anchor_path, candidate_paths, "anchor", self.similarity_threshold)

        candidates: list[CandidateResult] = []
        for cpath in candidate_paths:
            cand_emb = self._runtime.extract_embedding(cpath)
            if cand_emb is None:
                candidates.append(
                    CandidateResult(
                        embedding=EmbeddingScore(
                            cosine_similarity=0.0,
                            threshold=self.similarity_threshold,
                            passed=False,
                        ),
                        composite_score=0.0,
                        ok=False,
                        failure_reasons=["no_face_detected_in_candidate"],
                    )
                )
                continue

            sim = _cosine_similarity(anchor_emb, cand_emb)
            passed = sim >= self.similarity_threshold
            candidates.append(
                CandidateResult(
                    embedding=EmbeddingScore(
                        cosine_similarity=round(sim, 6),
                        threshold=self.similarity_threshold,
                        passed=passed,
                    ),
                    composite_score=round(sim, 6),
                    ok=passed,
                    failure_reasons=[] if passed else ["embedding_below_threshold"],
                )
            )

        return EvaluationResult(candidates=candidates, anchor_path=anchor_path)


def _no_face_result(
    anchor_path: str, candidate_paths: list[str], source: str, threshold: float,
) -> EvaluationResult:
    return EvaluationResult(
        candidates=[
            CandidateResult(
                embedding=EmbeddingScore(cosine_similarity=0.0, threshold=threshold, passed=False),
                composite_score=0.0,
                ok=False,
                failure_reasons=[f"no_face_detected_in_{source}"],
            )
            for _ in candidate_paths
        ],
        anchor_path=anchor_path,
    )
