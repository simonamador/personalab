"""Sharpness evaluation via Laplacian variance.

Measures high-frequency content as a proxy for image sharpness.
Each frame is compared against the anchor's Laplacian variance as a ratio
so that the metric is independent of resolution or scene complexity.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from personalab.quality.schemas import FrameSharpness

logger = logging.getLogger(__name__)

_LAPLACIAN_KSIZE = 3


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Return variance of the Laplacian of *image* (BGR or grayscale uint8).

    Higher values indicate more high-frequency content (sharper).
    Returns 0.0 on failure.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=_LAPLACIAN_KSIZE)
    return float(np.var(lap))


class SharpnessEvaluator:
    """Evaluate per-frame sharpness relative to an anchor image.

    The anchor's Laplacian variance is computed once and cached.
    Each frame's ratio ``frame_var / anchor_var`` is compared against
    ``min_sharpness_ratio``.
    """

    def __init__(self, *, min_sharpness_ratio: float = 0.4) -> None:
        self.min_sharpness_ratio = min_sharpness_ratio
        self._anchor_variance: float | None = None

    def set_anchor(self, anchor_image: np.ndarray) -> float:
        """Compute and cache the anchor's Laplacian variance. Returns the value."""
        self._anchor_variance = compute_laplacian_variance(anchor_image)
        if self._anchor_variance < 1e-6:
            logger.warning("Anchor Laplacian variance is near zero; sharpness ratios will be capped")
        return self._anchor_variance

    @property
    def anchor_variance(self) -> float:
        if self._anchor_variance is None:
            raise RuntimeError("Call set_anchor() before evaluating frames")
        return self._anchor_variance

    def evaluate_frame(self, frame: np.ndarray, frame_index: int) -> FrameSharpness:
        """Evaluate a single frame against the cached anchor variance."""
        frame_var = compute_laplacian_variance(frame)
        anchor_var = self.anchor_variance
        ratio = frame_var / anchor_var if anchor_var > 1e-6 else 0.0
        return FrameSharpness(
            frame_index=frame_index,
            laplacian_variance=round(frame_var, 4),
            anchor_ratio=round(ratio, 6),
        )
