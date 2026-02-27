"""Geometric (landmark-based) identity evaluator.

Uses 106-point landmarks when available (preferred) with fallback to 5-point.
Supports adaptive thresholds that relax when face detection confidence is low
or the face bounding box is small (e.g. full-body shots).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from personalab.identity.face_runtime import FaceRuntime
from personalab.identity.schemas import (
    CandidateResult,
    EvaluationResult,
    LandmarkScore,
)

logger = logging.getLogger(__name__)

# -- 106-point landmark indices (2d106det model) --------------------------
# Ratios chosen to capture distinct facial proportions:
_RATIO_NAMES_106 = [
    "inter_eye_to_nose_tip",
    "nose_length_to_inter_eye",
    "mouth_width_to_inter_eye",
    "jaw_width_to_inter_eye",
    "nose_width_to_inter_eye",
    "upper_lip_to_chin",
    "eye_height_to_inter_eye",
]

# Indices into the 106-point array (InsightFace 2d106det convention)
_IDX_LEFT_EYE_CENTER = 38
_IDX_RIGHT_EYE_CENTER = 88
_IDX_NOSE_TIP = 86
_IDX_NOSE_BRIDGE_TOP = 43
_IDX_MOUTH_LEFT = 52
_IDX_MOUTH_RIGHT = 61
_IDX_JAW_LEFT = 0
_IDX_JAW_RIGHT = 32
_IDX_NOSE_LEFT = 83
_IDX_NOSE_RIGHT = 87
_IDX_UPPER_LIP_CENTER = 63
_IDX_CHIN = 16
_IDX_LEFT_EYE_TOP = 37
_IDX_LEFT_EYE_BOTTOM = 40


def _compute_ratios_106(lm: np.ndarray) -> np.ndarray:
    """Compute 7 normalised ratios from 106-point landmarks."""
    left_eye = lm[_IDX_LEFT_EYE_CENTER]
    right_eye = lm[_IDX_RIGHT_EYE_CENTER]
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-6:
        return np.zeros(len(_RATIO_NAMES_106))

    nose_tip = lm[_IDX_NOSE_TIP]
    nose_bridge = lm[_IDX_NOSE_BRIDGE_TOP]
    mouth_left = lm[_IDX_MOUTH_LEFT]
    mouth_right = lm[_IDX_MOUTH_RIGHT]
    jaw_left = lm[_IDX_JAW_LEFT]
    jaw_right = lm[_IDX_JAW_RIGHT]
    nose_left = lm[_IDX_NOSE_LEFT]
    nose_right = lm[_IDX_NOSE_RIGHT]
    upper_lip = lm[_IDX_UPPER_LIP_CENTER]
    chin = lm[_IDX_CHIN]
    eye_top = lm[_IDX_LEFT_EYE_TOP]
    eye_bottom = lm[_IDX_LEFT_EYE_BOTTOM]

    return np.array([
        float(np.linalg.norm(nose_tip - (left_eye + right_eye) / 2)) / inter_eye,
        float(np.linalg.norm(nose_tip - nose_bridge)) / inter_eye,
        float(np.linalg.norm(mouth_right - mouth_left)) / inter_eye,
        float(np.linalg.norm(jaw_right - jaw_left)) / inter_eye,
        float(np.linalg.norm(nose_right - nose_left)) / inter_eye,
        float(np.linalg.norm(chin - upper_lip)) / inter_eye,
        float(np.linalg.norm(eye_bottom - eye_top)) / inter_eye,
    ])


# -- 5-point fallback -----------------------------------------------------
_RATIO_NAMES_5 = [
    "nose_to_mouth_center",
    "mouth_width_to_inter_eye",
]


def _compute_ratios_5(landmarks: np.ndarray) -> np.ndarray:
    """Fallback: 2 unique ratios from 5-point landmarks."""
    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-6:
        return np.zeros(len(_RATIO_NAMES_5))
    mouth_center = (mouth_left + mouth_right) / 2.0
    nose_to_mouth = float(np.linalg.norm(nose - mouth_center))
    mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
    return np.array([nose_to_mouth / inter_eye, mouth_width / inter_eye])


# Exported for tests
RATIO_NAMES_106 = _RATIO_NAMES_106
RATIO_NAMES_5 = _RATIO_NAMES_5

# Adaptive threshold constants
_MIN_BBOX_FRACTION = 0.10
_MAX_BBOX_FRACTION = 0.30
_RELAXATION_FACTOR = 2.5


def _adaptive_max_error(
    base_max_error: float,
    face_bbox: np.ndarray | None,
    image_path: str,
) -> float:
    """Relax the geometric threshold for small faces (full-body shots).

    When the face bbox occupies less than ``_MIN_BBOX_FRACTION`` of the image
    diagonal, the threshold is multiplied by up to ``_RELAXATION_FACTOR``.
    """
    if face_bbox is None:
        return base_max_error * _RELAXATION_FACTOR

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return base_max_error

    img_h, img_w = img.shape[:2]
    img_diag = np.sqrt(img_h ** 2 + img_w ** 2)
    if img_diag < 1e-6:
        return base_max_error

    x1, y1, x2, y2 = face_bbox
    face_diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    ratio = float(face_diag / img_diag)

    if ratio >= _MAX_BBOX_FRACTION:
        return base_max_error
    if ratio <= _MIN_BBOX_FRACTION:
        return base_max_error * _RELAXATION_FACTOR

    t = (_MAX_BBOX_FRACTION - ratio) / (_MAX_BBOX_FRACTION - _MIN_BBOX_FRACTION)
    return base_max_error * (1.0 + t * (_RELAXATION_FACTOR - 1.0))


class GeometricEvaluator:
    """Evaluates identity consistency via landmark-derived facial ratios.

    Prefers 106-point landmarks for richer signal; falls back to 5-point.
    Adapts the error threshold based on face bounding-box size.
    """

    def __init__(
        self,
        *,
        runtime: FaceRuntime,
        max_normalized_error: float = 0.08,
    ) -> None:
        self._runtime = runtime
        self.max_normalized_error = max_normalized_error
        self._anchor_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def _get_ratios(self, image_path: str) -> tuple[np.ndarray | None, list[str]]:
        """Extract ratios + names using 106-point landmarks, falling back to 5-point."""
        lm106 = self._runtime.extract_landmarks_106(image_path)
        if lm106 is not None:
            return _compute_ratios_106(lm106), _RATIO_NAMES_106

        lm5 = self._runtime.extract_landmarks_5(image_path)
        if lm5 is not None:
            return _compute_ratios_5(lm5), _RATIO_NAMES_5

        return None, []

    def _get_anchor_ratios(self, anchor_path: str) -> tuple[np.ndarray | None, list[str]]:
        if anchor_path not in self._anchor_cache:
            ratios, names = self._get_ratios(anchor_path)
            if ratios is not None:
                self._anchor_cache[anchor_path] = (ratios, names)
        cached = self._anchor_cache.get(anchor_path)
        if cached is None:
            return None, []
        return cached

    def evaluate(self, anchor_path: str, candidate_paths: list[str]) -> EvaluationResult:
        anchor_ratios, anchor_names = self._get_anchor_ratios(anchor_path)
        if anchor_ratios is None:
            return _no_landmark_result(anchor_path, candidate_paths, "anchor", self.max_normalized_error)

        candidates: list[CandidateResult] = []
        for cpath in candidate_paths:
            cand_ratios, cand_names = self._get_ratios(cpath)
            if cand_ratios is None:
                candidates.append(
                    CandidateResult(
                        geometric=LandmarkScore(
                            normalized_error=1.0,
                            max_allowed=self.max_normalized_error,
                            passed=False,
                        ),
                        composite_score=0.0,
                        ok=False,
                        failure_reasons=["no_landmarks_in_candidate"],
                    )
                )
                continue

            shared_names = anchor_names if anchor_names == cand_names else _RATIO_NAMES_5
            shared_len = min(len(anchor_ratios), len(cand_ratios), len(shared_names))
            a = anchor_ratios[:shared_len]
            c = cand_ratios[:shared_len]
            names = shared_names[:shared_len]

            deltas = np.abs(a - c)
            mean_err = float(np.mean(deltas))

            face_bbox = self._runtime.face_bbox(cpath)
            effective_max = _adaptive_max_error(self.max_normalized_error, face_bbox, cpath)
            passed = mean_err <= effective_max

            landmark_deltas = {name: round(float(d), 6) for name, d in zip(names, deltas)}
            geo_score = max(0.0, 1.0 - mean_err)

            candidates.append(
                CandidateResult(
                    geometric=LandmarkScore(
                        normalized_error=round(mean_err, 6),
                        max_allowed=round(effective_max, 6),
                        passed=passed,
                        landmark_deltas=landmark_deltas,
                    ),
                    composite_score=round(geo_score, 6),
                    ok=passed,
                    failure_reasons=[] if passed else ["geometric_error_above_threshold"],
                )
            )

        return EvaluationResult(candidates=candidates, anchor_path=anchor_path)


def _no_landmark_result(
    anchor_path: str, candidate_paths: list[str], source: str, max_err: float,
) -> EvaluationResult:
    return EvaluationResult(
        candidates=[
            CandidateResult(
                geometric=LandmarkScore(
                    normalized_error=1.0, max_allowed=max_err, passed=False,
                ),
                composite_score=0.0,
                ok=False,
                failure_reasons=[f"no_landmarks_in_{source}"],
            )
            for _ in candidate_paths
        ],
        anchor_path=anchor_path,
    )
