"""Tests for the composed VideoQualityEvaluator."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from personalab.identity.frame_extractor import ExtractedFrame
from personalab.quality.illumination import IlluminationEvaluator
from personalab.quality.schemas import VideoQualityResult
from personalab.quality.sharpness import SharpnessEvaluator
from personalab.quality.video_quality_evaluator import VideoQualityEvaluator


def _make_anchor(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "anchor.png"
    cv2.imwrite(str(path), img)
    return str(path)


def _make_frames(n: int = 4, *, noisy: bool = True) -> list[ExtractedFrame]:
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n):
        if noisy:
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        else:
            img = np.full((64, 64, 3), 128 + i, dtype=np.uint8)
        frames.append(ExtractedFrame(image=img, frame_index=i * 10, timestamp_s=i * 0.5))
    return frames


class TestVideoQualityEvaluator:
    def test_evaluate_returns_correct_structure(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        frames = _make_frames(4, noisy=True)

        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(min_sharpness_ratio=0.4),
            illumination=IlluminationEvaluator(max_histogram_distance=0.35),
        )
        result = ev.evaluate(anchor, frames)

        assert isinstance(result, VideoQualityResult)
        assert len(result.frame_sharpness) == 4
        assert len(result.illumination_checks) == 3
        assert result.mean_sharpness_ratio > 0.0
        assert result.min_sharpness_ratio > 0.0

    def test_noisy_frames_pass_sharpness(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        frames = _make_frames(4, noisy=True)

        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(min_sharpness_ratio=0.3),
            illumination=IlluminationEvaluator(max_histogram_distance=0.5),
        )
        result = ev.evaluate(anchor, frames)
        assert result.sharpness_ok is True

    def test_uniform_frames_fail_sharpness(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        frames = _make_frames(4, noisy=False)

        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(min_sharpness_ratio=0.4),
            illumination=IlluminationEvaluator(max_histogram_distance=0.5),
        )
        result = ev.evaluate(anchor, frames)
        assert result.sharpness_ok is False
        assert result.min_sharpness_ratio < 0.1

    def test_stable_illumination_passes(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        frames = [
            ExtractedFrame(image=img.copy(), frame_index=i, timestamp_s=i * 0.1)
            for i in range(3)
        ]

        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(min_sharpness_ratio=0.0),
            illumination=IlluminationEvaluator(max_histogram_distance=0.35),
        )
        result = ev.evaluate(anchor, frames)
        assert result.illumination_ok is True
        assert result.max_histogram_distance == pytest.approx(0.0, abs=1e-4)

    def test_single_frame_has_no_illumination_checks(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        frames = _make_frames(1, noisy=True)

        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(min_sharpness_ratio=0.1),
            illumination=IlluminationEvaluator(),
        )
        result = ev.evaluate(anchor, frames)
        assert len(result.illumination_checks) == 0
        assert result.illumination_ok is True

    def test_empty_frames_returns_defaults(self, tmp_path):
        anchor = _make_anchor(tmp_path)
        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(),
            illumination=IlluminationEvaluator(),
        )
        result = ev.evaluate(anchor, [])
        assert result.mean_sharpness_ratio == 0.0
        assert result.illumination_ok is True

    def test_bad_anchor_path_returns_defaults(self):
        ev = VideoQualityEvaluator(
            sharpness=SharpnessEvaluator(),
            illumination=IlluminationEvaluator(),
        )
        result = ev.evaluate("/nonexistent/anchor.png", _make_frames(2))
        assert isinstance(result, VideoQualityResult)
        assert result.mean_sharpness_ratio == 0.0
