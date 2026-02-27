"""Tests for IdentityEvaluator protocol, StubEvaluator, and CompositeEvaluator."""

from personalab.identity.evaluator import StubEvaluator
from personalab.identity.composite_evaluator import CompositeEvaluator
from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
    LandmarkScore,
)


class TestStubEvaluator:
    def test_returns_evaluation_result(self):
        ev = StubEvaluator()
        result = ev.evaluate("/anchor.png", ["/cand1.png", "/cand2.png"])
        assert isinstance(result, EvaluationResult)
        assert len(result.candidates) == 2

    def test_all_candidates_pass(self):
        ev = StubEvaluator()
        result = ev.evaluate("/anchor.png", ["/c.png"])
        cand = result.first()
        assert cand.ok is True
        assert cand.composite_score == 1.0
        assert cand.embedding is not None
        assert cand.embedding.passed is True
        assert cand.failure_reasons == []

    def test_anchor_path_preserved(self):
        ev = StubEvaluator()
        result = ev.evaluate("/my/anchor.png", ["/c.png"])
        assert result.anchor_path == "/my/anchor.png"

    def test_empty_candidates(self):
        ev = StubEvaluator()
        result = ev.evaluate("/anchor.png", [])
        assert result.candidates == []


class _FakeEmbeddingEvaluator:
    """Returns controllable embedding scores for testing CompositeEvaluator."""

    def __init__(self, similarities: list[float], threshold: float = 0.5):
        self._sims = similarities
        self._threshold = threshold

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        candidates = []
        for i, cpath in enumerate(candidate_paths):
            sim = self._sims[i] if i < len(self._sims) else 0.0
            passed = sim >= self._threshold
            candidates.append(
                CandidateResult(
                    embedding=EmbeddingScore(
                        cosine_similarity=sim, threshold=self._threshold, passed=passed,
                    ),
                    composite_score=sim,
                    ok=passed,
                    failure_reasons=[] if passed else ["embedding_below_threshold"],
                )
            )
        return EvaluationResult(candidates=candidates, anchor_path=anchor_path)


class _FakeGeometricEvaluator:
    """Returns controllable geometric scores for testing CompositeEvaluator."""

    def __init__(self, errors: list[float], max_err: float = 0.08):
        self._errors = errors
        self._max_err = max_err

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        candidates = []
        for i, cpath in enumerate(candidate_paths):
            err = self._errors[i] if i < len(self._errors) else 1.0
            passed = err <= self._max_err
            candidates.append(
                CandidateResult(
                    geometric=LandmarkScore(
                        normalized_error=err, max_allowed=self._max_err, passed=passed,
                    ),
                    composite_score=max(0.0, 1.0 - err),
                    ok=passed,
                    failure_reasons=[] if passed else ["geometric_error_above_threshold"],
                )
            )
        return EvaluationResult(candidates=candidates, anchor_path=anchor_path)


class TestCompositeEvaluator:
    def test_embedding_only(self):
        emb = _FakeEmbeddingEvaluator([0.8])
        comp = CompositeEvaluator(embedding_evaluator=emb, geometric_evaluator=None)
        result = comp.evaluate("/anchor.png", ["/cand.png"])
        cand = result.first()
        assert cand.composite_score == 0.8
        assert cand.ok is True
        assert cand.embedding is not None
        assert cand.geometric is None

    def test_weighted_composite(self):
        emb = _FakeEmbeddingEvaluator([0.7], threshold=0.5)
        geo = _FakeGeometricEvaluator([0.05], max_err=0.08)
        comp = CompositeEvaluator(
            embedding_evaluator=emb,
            geometric_evaluator=geo,
            weight_embedding=0.7,
            weight_geometric=0.3,
        )
        result = comp.evaluate("/anchor.png", ["/cand.png"])
        cand = result.first()
        expected = round(0.7 * 0.7 + 0.3 * (1.0 - 0.05), 6)
        assert cand.composite_score == expected
        assert cand.ok is True

    def test_both_fail(self):
        emb = _FakeEmbeddingEvaluator([0.2], threshold=0.5)
        geo = _FakeGeometricEvaluator([0.5], max_err=0.08)
        comp = CompositeEvaluator(
            embedding_evaluator=emb, geometric_evaluator=geo,
        )
        result = comp.evaluate("/anchor.png", ["/cand.png"])
        cand = result.first()
        assert cand.ok is False
        assert "embedding_below_threshold" in cand.failure_reasons
        assert "geometric_error_above_threshold" in cand.failure_reasons

    def test_multiple_candidates(self):
        emb = _FakeEmbeddingEvaluator([0.9, 0.3], threshold=0.5)
        comp = CompositeEvaluator(embedding_evaluator=emb)
        result = comp.evaluate("/anchor.png", ["/a.png", "/b.png"])
        assert len(result.candidates) == 2
        assert result.candidates[0].ok is True
        assert result.candidates[1].ok is False
