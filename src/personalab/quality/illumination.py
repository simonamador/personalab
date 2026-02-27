"""Illumination consistency evaluation via histogram distance.

Compares normalised grayscale histograms between key frames to detect
flicker or tonal drift that may indicate rendering instability.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from personalab.quality.schemas import IlluminationCheck

logger = logging.getLogger(__name__)

_HIST_SIZE = 256
_HIST_RANGE = [0, 256]

_CV_METHODS = {
    "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    "chi_squared": cv2.HISTCMP_CHISQR_ALT,
    "correlation": cv2.HISTCMP_CORREL,
}


def _normalised_hist(image: np.ndarray) -> np.ndarray:
    """Return a normalised 256-bin grayscale histogram."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [_HIST_SIZE], _HIST_RANGE)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def compute_histogram_distance(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "bhattacharyya",
) -> float:
    """Distance between two images' normalised grayscale histograms.

    For Bhattacharyya the result is in [0, 1]; for chi-squared it is
    unbounded but normalised variants are used.
    """
    cv_method = _CV_METHODS.get(method)
    if cv_method is None:
        raise ValueError(f"Unknown method {method!r}; choose from {list(_CV_METHODS)}")

    hist_a = _normalised_hist(a)
    hist_b = _normalised_hist(b)
    dist = cv2.compareHist(hist_a, hist_b, cv_method)

    if method == "correlation":
        return round(1.0 - float(dist), 6)
    return round(float(dist), 6)


class IlluminationEvaluator:
    """Evaluate illumination consistency between consecutive key frames."""

    def __init__(
        self,
        *,
        max_histogram_distance: float = 0.35,
        method: str = "bhattacharyya",
    ) -> None:
        if method not in _CV_METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {list(_CV_METHODS)}")
        self.max_histogram_distance = max_histogram_distance
        self.method = method

    def evaluate_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        idx_a: int,
        idx_b: int,
    ) -> IlluminationCheck:
        """Compare two frames and return an ``IlluminationCheck``."""
        dist = compute_histogram_distance(frame_a, frame_b, method=self.method)
        return IlluminationCheck(
            frame_index_a=idx_a,
            frame_index_b=idx_b,
            histogram_distance=dist,
            method=self.method,
        )
