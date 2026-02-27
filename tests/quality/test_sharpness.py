"""Tests for sharpness evaluation via Laplacian variance."""

import numpy as np
import pytest

from personalab.quality.sharpness import SharpnessEvaluator, compute_laplacian_variance
from personalab.quality.schemas import FrameSharpness


class TestComputeLaplacianVariance:
    def test_uniform_image_has_low_variance(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        var = compute_laplacian_variance(img)
        assert var == pytest.approx(0.0, abs=1.0)

    def test_noisy_image_has_high_variance(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        var = compute_laplacian_variance(img)
        assert var > 100.0

    def test_grayscale_input_works(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        var = compute_laplacian_variance(gray)
        assert var == pytest.approx(0.0, abs=1.0)

    def test_empty_image_returns_zero(self):
        assert compute_laplacian_variance(np.array([])) == 0.0

    def test_none_returns_zero(self):
        assert compute_laplacian_variance(None) == 0.0


class TestSharpnessEvaluator:
    def test_set_anchor_caches_variance(self):
        rng = np.random.default_rng(0)
        anchor = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ev = SharpnessEvaluator(min_sharpness_ratio=0.4)
        val = ev.set_anchor(anchor)
        assert val > 0.0
        assert ev.anchor_variance == val

    def test_evaluate_frame_returns_correct_structure(self):
        rng = np.random.default_rng(0)
        anchor = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        ev = SharpnessEvaluator(min_sharpness_ratio=0.4)
        ev.set_anchor(anchor)
        result = ev.evaluate_frame(frame, frame_index=5)

        assert isinstance(result, FrameSharpness)
        assert result.frame_index == 5
        assert result.laplacian_variance > 0.0
        assert result.anchor_ratio > 0.0

    def test_uniform_frame_against_sharp_anchor_has_low_ratio(self):
        rng = np.random.default_rng(0)
        anchor = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)

        ev = SharpnessEvaluator(min_sharpness_ratio=0.4)
        ev.set_anchor(anchor)
        result = ev.evaluate_frame(frame, frame_index=0)

        assert result.anchor_ratio < 0.1

    def test_raises_without_set_anchor(self):
        ev = SharpnessEvaluator()
        with pytest.raises(RuntimeError, match="set_anchor"):
            ev.evaluate_frame(np.zeros((64, 64, 3), dtype=np.uint8), frame_index=0)

    def test_uniform_anchor_gives_zero_ratio(self):
        anchor = np.full((64, 64, 3), 128, dtype=np.uint8)
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)

        ev = SharpnessEvaluator()
        ev.set_anchor(anchor)
        result = ev.evaluate_frame(frame, frame_index=0)
        assert result.anchor_ratio == pytest.approx(0.0, abs=0.01)
