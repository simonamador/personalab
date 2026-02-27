"""Tests for IdentityPolicy score-based decisions."""

from personalab.identity.identity_policy import IdentityPolicy, Decision
from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
    LandmarkScore,
)


def _make_result(score: float, ok: bool = True) -> EvaluationResult:
    return EvaluationResult(
        candidates=[
            CandidateResult(
                embedding=EmbeddingScore(cosine_similarity=score, threshold=0.55, passed=ok),
                composite_score=score,
                ok=ok,
                failure_reasons=[] if ok else ["test_failure"],
            )
        ],
        anchor_path="/anchor.png",
    )


class TestIdentityPolicyCompositeOnly:
    """Default mode: decision depends only on composite_score."""

    def test_accept_high_score(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=3)
        d = policy.decide(1, _make_result(0.7, ok=True))
        assert d.action == "ACCEPT"
        assert d.score == 0.7

    def test_accept_at_threshold(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=3)
        d = policy.decide(1, _make_result(0.55, ok=True))
        assert d.action == "ACCEPT"

    def test_accept_even_when_ok_false(self):
        """composite_only mode ignores the ok flag."""
        policy = IdentityPolicy(accept_threshold=0.55, policy_mode="composite_only")
        d = policy.decide(1, _make_result(0.75, ok=False))
        assert d.action == "ACCEPT"

    def test_default_mode_is_composite_only(self):
        policy = IdentityPolicy()
        assert policy.policy_mode == "composite_only"


class TestIdentityPolicyStrictMode:
    """Strict mode: requires composite >= threshold AND ok == True."""

    def test_accept_when_both_pass(self):
        policy = IdentityPolicy(accept_threshold=0.55, policy_mode="strict")
        d = policy.decide(1, _make_result(0.7, ok=True))
        assert d.action == "ACCEPT"

    def test_retry_when_score_high_but_ok_false(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, policy_mode="strict")
        d = policy.decide(1, _make_result(0.75, ok=False))
        assert d.action == "RETRY"

    def test_reject_when_below_retry(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, policy_mode="strict")
        d = policy.decide(1, _make_result(0.2, ok=False))
        assert d.action == "REJECT_FINAL"


class TestIdentityPolicyRetry:
    def test_retry_between_thresholds(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=3)
        d = policy.decide(1, _make_result(0.45, ok=False))
        assert d.action == "RETRY"
        assert "policy_overrides" in d.patches
        assert d.patches["policy_overrides"]["identity_strength"] == "high"

    def test_retry_escalation(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=5)
        d1 = policy.decide(1, _make_result(0.45, ok=False))
        d2 = policy.decide(2, _make_result(0.45, ok=False))
        d3 = policy.decide(3, _make_result(0.45, ok=False))
        assert d1.patches["policy_overrides"]["identity_strength"] == "high"
        assert d2.patches["policy_overrides"]["identity_strength"] == "very_high"
        assert d3.patches["policy_overrides"]["identity_strength"] == "maximum"


class TestIdentityPolicyReject:
    def test_reject_max_attempts(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=2)
        d = policy.decide(2, _make_result(0.45, ok=False))
        assert d.action == "REJECT_FINAL"
        assert "max_attempts_reached" in d.notes

    def test_reject_below_retry_threshold(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=5)
        d = policy.decide(1, _make_result(0.2, ok=False))
        assert d.action == "REJECT_FINAL"
        assert "score_below_retry_threshold" in d.notes


class TestIdentityPolicyFromConfig:
    def test_from_config(self):
        cfg = {
            "scoring": {"accept_threshold": 0.6, "retry_threshold": 0.4, "policy_mode": "strict"},
            "retry": {"max_attempts": 5},
        }
        policy = IdentityPolicy.from_config(cfg)
        assert policy.accept_threshold == 0.6
        assert policy.retry_threshold == 0.4
        assert policy.max_attempts == 5
        assert policy.policy_mode == "strict"

    def test_from_config_defaults(self):
        policy = IdentityPolicy.from_config({})
        assert policy.accept_threshold == 0.55
        assert policy.retry_threshold == 0.35
        assert policy.max_attempts == 3
        assert policy.policy_mode == "composite_only"


class TestIdentityPolicyLegacyDict:
    def test_dict_ok_true(self):
        policy = IdentityPolicy(accept_threshold=0.55)
        d = policy.decide(1, {"ok": True})
        assert d.action == "ACCEPT"

    def test_dict_ok_false(self):
        policy = IdentityPolicy(accept_threshold=0.55, retry_threshold=0.35, max_attempts=3)
        d = policy.decide(1, {"ok": False, "failure_reasons": ["face_mismatch"]})
        assert d.action in ("RETRY", "REJECT_FINAL")


class TestDecisionModel:
    def test_score_field(self):
        d = Decision(action="ACCEPT", score=0.8)
        assert d.score == 0.8

    def test_default_score(self):
        d = Decision(action="ACCEPT")
        assert d.score == 0.0
