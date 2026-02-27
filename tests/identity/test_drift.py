"""Tests for DriftTracker: append, history, and summary."""

import json

import pytest

from personalab.identity.drift import DriftTracker, _compute_trend
from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
    DriftRecord,
)


def _make_eval_result(score: float = 0.7, sim: float = 0.75) -> EvaluationResult:
    return EvaluationResult(
        candidates=[
            CandidateResult(
                embedding=EmbeddingScore(cosine_similarity=sim, threshold=0.5, passed=True),
                composite_score=score,
                ok=True,
            )
        ],
        anchor_path="/anchor.png",
    )


class TestDriftTracker:
    def test_record_creates_file(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        tracker.record("persona_a", _make_eval_result(), generation_id="gen1")
        path = tmp_path / "persona_a_drift.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["persona"] == "persona_a"
        assert data["generation_id"] == "gen1"
        assert data["composite_score"] == 0.7

    def test_record_appends(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        tracker.record("p", _make_eval_result(0.6), generation_id="g1")
        tracker.record("p", _make_eval_result(0.8), generation_id="g2")
        lines = (tmp_path / "p_drift.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_history_returns_all(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        for i in range(5):
            tracker.record("p", _make_eval_result(0.5 + i * 0.1), generation_id=f"g{i}")
        recs = tracker.history("p")
        assert len(recs) == 5
        assert all(isinstance(r, DriftRecord) for r in recs)

    def test_history_with_window(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        for i in range(10):
            tracker.record("p", _make_eval_result(0.5), generation_id=f"g{i}")
        recs = tracker.history("p", window=3)
        assert len(recs) == 3

    def test_history_empty_persona(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        assert tracker.history("nonexistent") == []

    def test_record_with_candidate_result(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        cand = CandidateResult(
            embedding=EmbeddingScore(cosine_similarity=0.9, threshold=0.5, passed=True),
            composite_score=0.85,
            ok=True,
        )
        rec = tracker.record("p", cand, generation_id="g1", decision_action="ACCEPT")
        assert rec.decision_action == "ACCEPT"
        assert rec.composite_score == 0.85

    def test_record_returns_drift_record(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        rec = tracker.record("p", _make_eval_result(), generation_id="g1")
        assert isinstance(rec, DriftRecord)
        assert rec.persona == "p"

    def test_summary_empty(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        s = tracker.summary("empty")
        assert s.count == 0
        assert s.trend == "stable"

    def test_summary_basic(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        for score in [0.5, 0.6, 0.7, 0.8, 0.9]:
            tracker.record("p", _make_eval_result(score))
        s = tracker.summary("p")
        assert s.count == 5
        assert s.min_score == 0.5
        assert s.max_score == 0.9
        assert s.trend == "improving"

    def test_summary_degrading(self, tmp_path):
        tracker = DriftTracker(log_dir=tmp_path)
        for score in [0.9, 0.8, 0.7, 0.6, 0.5]:
            tracker.record("p", _make_eval_result(score))
        s = tracker.summary("p")
        assert s.trend == "degrading"


class TestComputeTrend:
    def test_improving(self):
        assert _compute_trend([0.3, 0.5, 0.7, 0.9]) == "improving"

    def test_degrading(self):
        assert _compute_trend([0.9, 0.7, 0.5, 0.3]) == "degrading"

    def test_stable(self):
        assert _compute_trend([0.5, 0.5, 0.5, 0.5]) == "stable"

    def test_too_few_samples(self):
        assert _compute_trend([0.5, 0.9]) == "stable"
