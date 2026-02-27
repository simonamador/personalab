"""Composed video quality evaluator: sharpness + illumination consistency.

Operates on already-extracted frames and an anchor image.
Completely independent of the identity evaluation pipeline.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from personalab.identity.frame_extractor import ExtractedFrame
from personalab.quality.illumination import IlluminationEvaluator
from personalab.quality.schemas import (
    FrameSharpness,
    IlluminationCheck,
    VideoQualityResult,
)
from personalab.quality.sharpness import SharpnessEvaluator

logger = logging.getLogger(__name__)


class VideoQualityEvaluator:
    """Evaluate visual quality of video frames against an anchor.

    Composes ``SharpnessEvaluator`` (Laplacian variance ratio) and
    ``IlluminationEvaluator`` (histogram distance between consecutive
    key frames).  Both sub-evaluators are injected.
    """

    def __init__(
        self,
        *,
        sharpness: SharpnessEvaluator,
        illumination: IlluminationEvaluator,
    ) -> None:
        self._sharpness = sharpness
        self._illumination = illumination

    def evaluate(
        self,
        anchor_path: str,
        frames: list[ExtractedFrame],
    ) -> VideoQualityResult:
        """Run sharpness + illumination checks and aggregate results."""
        anchor_img = cv2.imread(anchor_path)
        if anchor_img is None:
            logger.warning("Could not read anchor image at %s", anchor_path)
            return VideoQualityResult()

        self._sharpness.set_anchor(anchor_img)

        sharpness_results = self._evaluate_sharpness(frames)
        illumination_results = self._evaluate_illumination(frames)

        return self._aggregate(sharpness_results, illumination_results)

    def _evaluate_sharpness(
        self, frames: list[ExtractedFrame],
    ) -> list[FrameSharpness]:
        results: list[FrameSharpness] = []
        for ef in frames:
            results.append(
                self._sharpness.evaluate_frame(ef.image, ef.frame_index),
            )
        return results

    def _evaluate_illumination(
        self, frames: list[ExtractedFrame],
    ) -> list[IlluminationCheck]:
        if len(frames) < 2:
            return []
        checks: list[IlluminationCheck] = []
        for i in range(len(frames) - 1):
            checks.append(
                self._illumination.evaluate_pair(
                    frames[i].image,
                    frames[i + 1].image,
                    idx_a=frames[i].frame_index,
                    idx_b=frames[i + 1].frame_index,
                ),
            )
        return checks

    def _aggregate(
        self,
        sharpness: list[FrameSharpness],
        illumination: list[IlluminationCheck],
    ) -> VideoQualityResult:
        if sharpness:
            ratios = np.array([s.anchor_ratio for s in sharpness], dtype=np.float64)
            mean_r = round(float(np.mean(ratios)), 6)
            min_r = round(float(np.min(ratios)), 6)
            std_r = round(float(np.std(ratios, ddof=0)), 6)
            sharpness_ok = bool(min_r >= self._sharpness.min_sharpness_ratio)
        else:
            mean_r = min_r = std_r = 0.0
            sharpness_ok = True

        if illumination:
            dists = np.array(
                [c.histogram_distance for c in illumination], dtype=np.float64,
            )
            mean_d = round(float(np.mean(dists)), 6)
            max_d = round(float(np.max(dists)), 6)
            illumination_ok = bool(max_d <= self._illumination.max_histogram_distance)
        else:
            mean_d = max_d = 0.0
            illumination_ok = True

        return VideoQualityResult(
            mean_sharpness_ratio=mean_r,
            min_sharpness_ratio=min_r,
            std_sharpness_ratio=std_r,
            frame_sharpness=sharpness,
            sharpness_ok=sharpness_ok,
            illumination_checks=illumination,
            mean_histogram_distance=mean_d,
            max_histogram_distance=max_d,
            illumination_ok=illumination_ok,
        )
