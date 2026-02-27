"""Tests for GenerationMetricsLogger structured JSONL output."""

import json

import pytest

from personalab.observability.metrics_logger import GenerationMetricsLogger
from personalab.identity.schemas import (
    CandidateResult,
    EmbeddingScore,
    EvaluationResult,
    LandmarkScore,
)


def _make_eval(score: float = 0.75, sim: float = 0.8, geo_err: float = 0.04) -> EvaluationResult:
    return EvaluationResult(
        candidates=[
            CandidateResult(
                embedding=EmbeddingScore(cosine_similarity=sim, threshold=0.55, passed=True),
                geometric=LandmarkScore(
                    normalized_error=geo_err, max_allowed=0.08, passed=True,
                ),
                composite_score=score,
                ok=True,
            )
        ],
        anchor_path="/anchor.png",
    )


class TestGenerationMetricsLogger:
    def test_log_creates_file(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(
            persona="test_persona",
            attempt=1,
            eval_result=_make_eval(),
            decision_action="ACCEPT",
            model_name="test-model",
        )
        path = tmp_path / "test_persona" / "metrics.jsonl"
        assert path.exists()
        assert record["persona"] == "test_persona"
        assert record["attempt"] == 1
        assert record["decision"] == "ACCEPT"

    def test_log_scores(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(
            persona="p",
            attempt=1,
            eval_result=_make_eval(score=0.75, sim=0.8, geo_err=0.04),
        )
        assert record["scores"]["composite_score"] == 0.75
        assert record["scores"]["embedding_similarity"] == 0.8
        assert record["scores"]["geometric_error"] == 0.04

    def test_log_without_eval_result(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(persona="p", attempt=1)
        assert record["scores"] == {}

    def test_log_timing(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(
            persona="p",
            attempt=1,
            generation_latency_ms=1234.5,
            evaluation_latency_ms=56.7,
        )
        assert record["timing"]["generation_latency_ms"] == 1234.5
        assert record["timing"]["evaluation_latency_ms"] == 56.7

    def test_log_appends(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        logger.log(persona="p", attempt=1)
        logger.log(persona="p", attempt=2)
        path = tmp_path / "p" / "metrics.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_read_log(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        logger.log(persona="p", attempt=1, decision_action="RETRY")
        logger.log(persona="p", attempt=2, decision_action="ACCEPT")
        records = logger.read_log("p")
        assert len(records) == 2
        assert records[0]["decision"] == "RETRY"
        assert records[1]["decision"] == "ACCEPT"

    def test_read_log_last_n(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        for i in range(5):
            logger.log(persona="p", attempt=i)
        records = logger.read_log("p", last_n=2)
        assert len(records) == 2

    def test_read_log_empty(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        assert logger.read_log("nonexistent") == []

    def test_log_generation_id(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(persona="p", attempt=1, generation_id="abc123")
        assert record["generation_id"] == "abc123"

    def test_log_auto_generation_id(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(persona="p", attempt=1)
        assert len(record["generation_id"]) == 12

    def test_log_extra(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        record = logger.log(persona="p", attempt=1, extra={"custom": "data"})
        assert record["extra"]["custom"] == "data"

    def test_log_with_candidate_result(self, tmp_path):
        logger = GenerationMetricsLogger(output_dir=tmp_path)
        cand = CandidateResult(
            embedding=EmbeddingScore(cosine_similarity=0.9, threshold=0.5, passed=True),
            composite_score=0.9,
            ok=True,
        )
        record = logger.log(persona="p", attempt=1, eval_result=cand)
        assert record["scores"]["composite_score"] == 0.9
        assert record["scores"]["embedding_similarity"] == 0.9
