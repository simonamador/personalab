"""Tests for illumination consistency evaluation."""

import numpy as np
import pytest

from personalab.quality.illumination import (
    IlluminationEvaluator,
    compute_histogram_distance,
)
from personalab.quality.schemas import IlluminationCheck


class TestComputeHistogramDistance:
    def test_identical_images_have_zero_distance(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        dist = compute_histogram_distance(img, img)
        assert dist == pytest.approx(0.0, abs=1e-4)

    def test_different_images_have_nonzero_distance(self):
        a = np.full((64, 64, 3), 50, dtype=np.uint8)
        b = np.full((64, 64, 3), 200, dtype=np.uint8)
        dist = compute_histogram_distance(a, b)
        assert dist > 0.5

    def test_bhattacharyya_in_0_1_range(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        dist = compute_histogram_distance(a, b, method="bhattacharyya")
        assert 0.0 <= dist <= 1.0

    def test_chi_squared_method(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        dist = compute_histogram_distance(a, b, method="chi_squared")
        assert dist >= 0.0

    def test_correlation_method(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        dist = compute_histogram_distance(img, img, method="correlation")
        assert dist == pytest.approx(0.0, abs=1e-4)

    def test_unknown_method_raises(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_histogram_distance(img, img, method="invalid")

    def test_grayscale_input(self):
        a = np.full((64, 64), 100, dtype=np.uint8)
        b = np.full((64, 64), 100, dtype=np.uint8)
        dist = compute_histogram_distance(a, b)
        assert dist == pytest.approx(0.0, abs=1e-4)


class TestIlluminationEvaluator:
    def test_evaluate_pair_returns_correct_structure(self):
        a = np.full((64, 64, 3), 128, dtype=np.uint8)
        b = np.full((64, 64, 3), 130, dtype=np.uint8)
        ev = IlluminationEvaluator(max_histogram_distance=0.35)
        result = ev.evaluate_pair(a, b, idx_a=0, idx_b=5)

        assert isinstance(result, IlluminationCheck)
        assert result.frame_index_a == 0
        assert result.frame_index_b == 5
        assert result.method == "bhattacharyya"
        assert result.histogram_distance >= 0.0

    def test_identical_frames_have_low_distance(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        ev = IlluminationEvaluator()
        result = ev.evaluate_pair(img, img, idx_a=0, idx_b=1)
        assert result.histogram_distance == pytest.approx(0.0, abs=1e-4)

    def test_constructor_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            IlluminationEvaluator(method="bogus")
