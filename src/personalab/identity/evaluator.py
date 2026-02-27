"""Identity evaluator protocol and stub implementation.

IdentityEvaluator is a Protocol (structural typing, consistent with LLMClient).
StubEvaluator always passes -- use as fallback when insightface is not installed.
"""

from __future__ import annotations

from typing import Protocol

from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
)


class IdentityEvaluator(Protocol):
    """Contract for identity evaluation. Implementations compare candidates to an anchor."""

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult: ...


class StubEvaluator:
    """Always returns ok=True with composite_score=1.0. Drop-in when no real evaluator is available."""

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        return EvaluationResult(
            candidates=[
                CandidateResult(
                    embedding=EmbeddingScore(cosine_similarity=1.0, threshold=0.0, passed=True),
                    geometric=None,
                    composite_score=1.0,
                    ok=True,
                    failure_reasons=[],
                )
                for _ in candidate_paths
            ],
            anchor_path=anchor_path,
        )
