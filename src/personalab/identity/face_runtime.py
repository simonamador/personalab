"""Shared InsightFace runtime to avoid duplicate model loading.

Both ArcFaceEvaluator and GeometricEvaluator receive a FaceRuntime
via constructor injection, guaranteeing a single set of ONNX models in memory.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis

    _INSIGHTFACE_AVAILABLE = True
except ImportError:  # pragma: no cover
    FaceAnalysis = None  # type: ignore[assignment,misc]
    _INSIGHTFACE_AVAILABLE = False


def _onnx_providers(ctx_id: int) -> list[str]:
    if ctx_id >= 0:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceRuntime:
    """Centralised InsightFace model wrapper.

    Loads the model once and exposes face detection + embedding/landmark extraction
    that multiple evaluators share.
    """

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        if not _INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface is required for FaceRuntime. "
                "Install with: pip install personalab[eval]"
            )
        self._app = FaceAnalysis(name=model_name, providers=_onnx_providers(ctx_id))
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.model_name = model_name

    def get_faces(self, image_path: str) -> list[Any]:
        """Detect faces in *image_path*. Returns list of insightface Face objects."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Could not read image: %s", image_path)
            return []
        return self._app.get(img)

    def dominant_face(self, image_path: str) -> Any | None:
        """Return the largest face by bounding-box area, or None."""
        faces = self.get_faces(image_path)
        if not faces:
            logger.warning("No face detected in: %s", image_path)
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def extract_embedding(self, image_path: str) -> np.ndarray | None:
        """512-d ArcFace embedding from the dominant face, or None."""
        face = self.dominant_face(image_path)
        if face is None:
            return None
        emb = getattr(face, "embedding", None)
        return emb

    def extract_landmarks_5(self, image_path: str) -> np.ndarray | None:
        """5-point landmarks (left_eye, right_eye, nose, mouth_left, mouth_right) as (5,2), or None."""
        face = self.dominant_face(image_path)
        if face is None:
            return None
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            logger.warning("Landmarks unavailable for face in: %s", image_path)
            return None
        return np.array(kps[:5], dtype=np.float64)

    def extract_landmarks_106(self, image_path: str) -> np.ndarray | None:
        """106-point landmarks if available (from 2d106det model), or None."""
        face = self.dominant_face(image_path)
        if face is None:
            return None
        lm106 = getattr(face, "landmark_2d_106", None)
        if lm106 is None or len(lm106) < 106:
            return None
        return np.array(lm106[:106], dtype=np.float64)

    def face_bbox(self, image_path: str) -> np.ndarray | None:
        """Bounding box [x1, y1, x2, y2] of the dominant face, or None."""
        face = self.dominant_face(image_path)
        if face is None:
            return None
        return np.array(face.bbox, dtype=np.float64)

    def face_det_score(self, image_path: str) -> float:
        """Detection confidence of the dominant face (0.0 if no face)."""
        face = self.dominant_face(image_path)
        if face is None:
            return 0.0
        return float(getattr(face, "det_score", 0.0))
