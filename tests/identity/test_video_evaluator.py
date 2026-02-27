"""Tests for VideoIdentityEvaluator with mocked evaluator and frame extractor."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from personalab.identity.evaluator import StubEvaluator
from personalab.identity.frame_extractor import FrameExtractor, ExtractedFrame
from personalab.identity.video_evaluator import VideoIdentityEvaluator, StubVideoEvaluator
from personalab.identity.schemas import (
    VideoEvaluationResult,
    FrameEvaluationResult,
    CandidateResult,
    EmbeddingScore,
    EmbeddingTemporalStats,
    EvaluationResult,
)


def _make_synthetic_video(n_frames: int = 10) -> bytes:
    """Create a minimal MP4 video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (64, 64))
    for i in range(n_frames):
        frame = np.full((64, 64, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


def _make_anchor(tmp_path: Path) -> str:
    """Create a fake anchor PNG and return its path."""
    anchor = tmp_path / "anchor.png"
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(anchor), img)
    return str(anchor)


class TestVideoIdentityEvaluator:
    def test_evaluates_all_extracted_frames(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        video_data = _make_synthetic_video(n_frames=20)

        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        result = vid_eval.evaluate(anchor, video_data, max_frames=4, strategy="uniform")

        assert isinstance(result, VideoEvaluationResult)
        assert result.frames_evaluated == 4
        assert result.mean_composite_score == pytest.approx(1.0)
        assert result.min_composite_score == pytest.approx(1.0)
        assert len(result.frame_results) == 4

    def test_frame_results_have_correct_structure(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        video_data = _make_synthetic_video(n_frames=10)

        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        result = vid_eval.evaluate(anchor, video_data, max_frames=3)

        for fr in result.frame_results:
            assert isinstance(fr, FrameEvaluationResult)
            assert fr.frame_index >= 0
            assert fr.timestamp_s >= 0.0
            assert isinstance(fr.eval_result, CandidateResult)
            assert fr.eval_result.composite_score >= 0.0

    def test_empty_video_returns_zero_frames(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        result = vid_eval.evaluate(anchor, b"", max_frames=4)
        assert result.frames_evaluated == 0
        assert result.frame_results == []

    def test_temporal_stats_populated(self, tmp_path):
        """Embedding and composite temporal stats are computed for stub evaluator."""
        anchor = _make_anchor(tmp_path)
        video_data = _make_synthetic_video(n_frames=20)

        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        result = vid_eval.evaluate(anchor, video_data, max_frames=4)

        assert result.std_composite_score == pytest.approx(0.0)
        assert result.embedding_stats is not None
        assert isinstance(result.embedding_stats, EmbeddingTemporalStats)
        assert result.embedding_stats.mean_cosine == pytest.approx(1.0)
        assert result.embedding_stats.std_cosine == pytest.approx(0.0)
        assert result.embedding_stats.min_cosine == pytest.approx(1.0)

    def test_evaluate_frames_matches_evaluate(self, tmp_path):
        """evaluate_frames produces the same result as evaluate on the same frames."""
        anchor = _make_anchor(tmp_path)
        video_data = _make_synthetic_video(n_frames=10)

        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        frames = frame_extractor.extract(video_data, max_frames=3)
        result_a = vid_eval.evaluate(anchor, video_data, max_frames=3)
        result_b = vid_eval.evaluate_frames(anchor, frames)

        assert result_a.frames_evaluated == result_b.frames_evaluated
        assert result_a.mean_composite_score == pytest.approx(result_b.mean_composite_score)

    def test_empty_frames_returns_none_stats(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        result = vid_eval.evaluate_frames(anchor, [])
        assert result.embedding_stats is None
        assert result.geometric_stats is None
        assert result.std_composite_score == 0.0

    def test_as_evaluation_result_conversion(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        video_data = _make_synthetic_video(n_frames=10)

        evaluator = StubEvaluator()
        frame_extractor = FrameExtractor()
        vid_eval = VideoIdentityEvaluator(evaluator, frame_extractor)

        video_result = vid_eval.evaluate(anchor, video_data, max_frames=3)
        eval_result = video_result.as_evaluation_result()

        assert isinstance(eval_result, EvaluationResult)
        assert len(eval_result.candidates) == 1
        assert eval_result.candidates[0].composite_score == pytest.approx(1.0)
        assert eval_result.anchor_path == anchor


class TestStubVideoEvaluator:
    def test_returns_perfect_scores(self):
        stub = StubVideoEvaluator()
        result = stub.evaluate("anchor.png", b"video-data", max_frames=4)
        assert isinstance(result, VideoEvaluationResult)
        assert result.mean_composite_score == 1.0
        assert result.frames_evaluated == 0
