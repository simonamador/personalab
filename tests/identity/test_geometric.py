"""Tests for geometric evaluator logic using synthetic landmark data."""

import numpy as np
import pytest

from personalab.identity.geometric_evaluator import (
    _compute_ratios_5,
    _compute_ratios_106,
    _adaptive_max_error,
    RATIO_NAMES_5,
    RATIO_NAMES_106,
)


class TestComputeRatios5:
    """Test 5-point fallback ratio computation."""

    @staticmethod
    def _make_landmarks(
        left_eye=(30.0, 50.0),
        right_eye=(70.0, 50.0),
        nose=(50.0, 70.0),
        mouth_left=(35.0, 90.0),
        mouth_right=(65.0, 90.0),
    ) -> np.ndarray:
        return np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float64)

    def test_returns_correct_number_of_ratios(self):
        lm = self._make_landmarks()
        ratios = _compute_ratios_5(lm)
        assert len(ratios) == len(RATIO_NAMES_5)

    def test_identical_landmarks_produce_same_ratios(self):
        lm = self._make_landmarks()
        r1 = _compute_ratios_5(lm)
        r2 = _compute_ratios_5(lm)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_scale_invariance(self):
        lm_small = self._make_landmarks()
        lm_big = lm_small * 2.5
        r_small = _compute_ratios_5(lm_small)
        r_big = _compute_ratios_5(lm_big)
        np.testing.assert_array_almost_equal(r_small, r_big, decimal=5)

    def test_different_faces_produce_different_ratios(self):
        lm_a = self._make_landmarks(mouth_left=(30.0, 90.0), mouth_right=(70.0, 90.0))
        lm_b = self._make_landmarks(mouth_left=(40.0, 90.0), mouth_right=(60.0, 90.0))
        r_a = _compute_ratios_5(lm_a)
        r_b = _compute_ratios_5(lm_b)
        assert not np.allclose(r_a, r_b)

    def test_zero_inter_eye_returns_zeros(self):
        lm = self._make_landmarks(left_eye=(50.0, 50.0), right_eye=(50.0, 50.0))
        ratios = _compute_ratios_5(lm)
        np.testing.assert_array_equal(ratios, np.zeros(len(RATIO_NAMES_5)))

    def test_no_duplicate_ratios(self):
        """5-point ratios should all be unique (unlike the old 3-ratio version)."""
        assert len(RATIO_NAMES_5) == 2
        assert len(set(RATIO_NAMES_5)) == len(RATIO_NAMES_5)


class TestComputeRatios106:
    """Test 106-point ratio computation."""

    @staticmethod
    def _make_106_landmarks() -> np.ndarray:
        """Create synthetic 106-point array with known positions."""
        lm = np.random.RandomState(42).rand(106, 2) * 200
        lm[38] = [80.0, 100.0]   # left eye center
        lm[88] = [120.0, 100.0]  # right eye center
        lm[86] = [100.0, 130.0]  # nose tip
        lm[43] = [100.0, 95.0]   # nose bridge top
        lm[52] = [85.0, 150.0]   # mouth left
        lm[61] = [115.0, 150.0]  # mouth right
        lm[0] = [50.0, 120.0]    # jaw left
        lm[32] = [150.0, 120.0]  # jaw right
        lm[83] = [92.0, 125.0]   # nose left
        lm[87] = [108.0, 125.0]  # nose right
        lm[63] = [100.0, 145.0]  # upper lip center
        lm[16] = [100.0, 180.0]  # chin
        lm[37] = [80.0, 95.0]    # left eye top
        lm[40] = [80.0, 105.0]   # left eye bottom
        return lm

    def test_returns_7_ratios(self):
        lm = self._make_106_landmarks()
        ratios = _compute_ratios_106(lm)
        assert len(ratios) == 7
        assert len(RATIO_NAMES_106) == 7

    def test_scale_invariance(self):
        lm = self._make_106_landmarks()
        r1 = _compute_ratios_106(lm)
        r2 = _compute_ratios_106(lm * 3.0)
        np.testing.assert_array_almost_equal(r1, r2, decimal=5)

    def test_all_ratios_are_unique(self):
        assert len(set(RATIO_NAMES_106)) == len(RATIO_NAMES_106)

    def test_zero_inter_eye_returns_zeros(self):
        lm = self._make_106_landmarks()
        lm[38] = lm[88]  # collapse eyes
        ratios = _compute_ratios_106(lm)
        np.testing.assert_array_equal(ratios, np.zeros(7))


class TestAdaptiveThreshold:
    """Test _adaptive_max_error relaxation logic."""

    def test_no_bbox_returns_relaxed(self):
        result = _adaptive_max_error(0.08, None, "fake.png")
        assert result == pytest.approx(0.08 * 2.5)

    def test_large_face_returns_base(self):
        bbox = np.array([100, 100, 400, 400], dtype=np.float64)
        # We can't easily pass a real image, so test the None-image path
        result = _adaptive_max_error(0.08, bbox, "nonexistent.png")
        assert result == 0.08  # cv2.imread returns None -> base
