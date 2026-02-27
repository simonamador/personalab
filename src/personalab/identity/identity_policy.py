"""Deterministic retry policy for identity preservation.

Supports two modes via ``policy_mode``:
  - ``composite_only`` (default): accept/reject based solely on composite_score.
  - ``strict``: requires composite_score >= threshold AND all sub-evaluator checks pass.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from personalab.identity.schemas import CandidateResult, EvaluationResult

PolicyMode = Literal["composite_only", "strict"]


class Decision(BaseModel):
    """Policy output: accept, retry with patches, or reject."""

    action: str = Field(..., description="ACCEPT | RETRY | REJECT_FINAL")
    patches: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    score: float = Field(default=0.0, description="Composite score that triggered this decision")


_ESCALATION = {
    1: "high",
    2: "very_high",
}


class IdentityPolicy:
    """Score-based retry policy for identity preservation.

    Parameters come from config.evaluation.scoring and config.evaluation.retry.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        accept_threshold: float = 0.55,
        retry_threshold: float = 0.35,
        policy_mode: PolicyMode = "composite_only",
    ) -> None:
        self.max_attempts = max_attempts
        self.accept_threshold = accept_threshold
        self.retry_threshold = retry_threshold
        self.policy_mode: PolicyMode = policy_mode

    @classmethod
    def from_config(cls, evaluation_cfg: dict[str, Any]) -> "IdentityPolicy":
        """Build from the evaluation section of ProjectConfig."""
        scoring = evaluation_cfg.get("scoring", {})
        retry = evaluation_cfg.get("retry", {})
        return cls(
            max_attempts=retry.get("max_attempts", 3),
            accept_threshold=scoring.get("accept_threshold", 0.55),
            retry_threshold=scoring.get("retry_threshold", 0.35),
            policy_mode=scoring.get("policy_mode", "composite_only"),
        )

    def decide(self, attempt: int, eval_result: EvaluationResult | CandidateResult | dict[str, Any]) -> Decision:
        """Decide accept/retry/reject based on composite score.

        *eval_result* can be:
        - EvaluationResult  (uses first candidate)
        - CandidateResult   (single candidate)
        - dict              (legacy: looks for 'composite_score' or 'ok')
        """
        score, ok, failures = self._extract(eval_result)

        should_accept = score >= self.accept_threshold
        if self.policy_mode == "strict":
            should_accept = should_accept and ok

        if should_accept:
            return Decision(action="ACCEPT", patches={}, notes="passed_all_checks", score=score)

        if attempt >= self.max_attempts:
            return Decision(
                action="REJECT_FINAL", patches={}, notes="max_attempts_reached", score=score,
            )

        if score < self.retry_threshold:
            return Decision(
                action="REJECT_FINAL",
                patches={},
                notes="score_below_retry_threshold;" + ";".join(failures),
                score=score,
            )

        strength = _ESCALATION.get(attempt, "maximum")
        patches = {
            "policy_overrides": {
                "identity_strength": strength,
                "face_constraints": "keep same person identity; avoid beautification; preserve natural asymmetry",
                "quality_constraints": "sharp face focus; avoid blur; avoid over/underexposure",
                "expression_constraints": "neutral-to-soft smile; avoid extreme expressions",
            }
        }
        return Decision(
            action="RETRY", patches=patches, notes=";".join(failures), score=score,
        )

    @staticmethod
    def _extract(result: EvaluationResult | CandidateResult | dict[str, Any]) -> tuple[float, bool, list[str]]:
        if isinstance(result, EvaluationResult):
            c = result.first()
            return c.composite_score, c.ok, c.failure_reasons
        if isinstance(result, CandidateResult):
            return result.composite_score, result.ok, result.failure_reasons
        # Legacy dict fallback
        return (
            float(result.get("composite_score", 1.0 if result.get("ok", False) else 0.0)),
            bool(result.get("ok", False)),
            list(result.get("failure_reasons", [])),
        )
