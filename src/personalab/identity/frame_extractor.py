"""Extract key frames from video bytes for identity evaluation."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFrame:
    """A single extracted frame with its index and timestamp."""
    image: np.ndarray
    frame_index: int
    timestamp_s: float


class FrameExtractor:
    """Extract key frames from video data using OpenCV.

    Strategies:
    - ``uniform``: sample *max_frames* evenly-spaced frames across the video.
    - ``face_detected``: iterate all frames, keep those where a face is detected
      (requires a ``FaceRuntime`` for detection). Falls back to uniform if no
      FaceRuntime is provided.
    """

    def __init__(self, face_runtime: object | None = None) -> None:
        self._face_runtime = face_runtime

    def extract(
        self,
        video_data: bytes,
        *,
        strategy: str = "uniform",
        max_frames: int = 8,
    ) -> list[ExtractedFrame]:
        """Extract key frames from raw video bytes.

        Returns a list of ``ExtractedFrame`` with BGR numpy arrays
        (same format as ``cv2.imread``).
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_data)
            tmp_path = f.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.error("Failed to open video from %d bytes", len(video_data))
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if total_frames <= 0:
                logger.warning("Video reports 0 frames")
                cap.release()
                return []

            if strategy == "face_detected" and self._face_runtime is not None:
                frames = self._extract_face_detected(cap, total_frames, fps, max_frames)
            else:
                if strategy == "face_detected" and self._face_runtime is None:
                    logger.warning(
                        "face_detected strategy requested but no FaceRuntime provided; "
                        "falling back to uniform"
                    )
                frames = self._extract_uniform(cap, total_frames, fps, max_frames)

            cap.release()
            return frames
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _extract_uniform(
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        max_frames: int,
    ) -> list[ExtractedFrame]:
        """Sample evenly-spaced frames across the video."""
        n = min(max_frames, total_frames)
        if n <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        frames: list[ExtractedFrame] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames.append(ExtractedFrame(
                image=frame,
                frame_index=int(idx),
                timestamp_s=float(idx) / fps,
            ))
        return frames

    def _extract_face_detected(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        max_frames: int,
    ) -> list[ExtractedFrame]:
        """Iterate frames and keep those where a face is detected."""
        step = max(1, total_frames // (max_frames * 4))
        frames: list[ExtractedFrame] = []
        idx = 0

        while idx < total_frames and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                idx += step
                continue

            if self._has_face(frame):
                frames.append(ExtractedFrame(
                    image=frame,
                    frame_index=idx,
                    timestamp_s=float(idx) / fps,
                ))
            idx += step

        if not frames:
            logger.warning("No faces detected in any sampled frame; falling back to uniform")
            return self._extract_uniform(cap, total_frames, fps, max_frames)

        return frames

    def _has_face(self, frame: np.ndarray) -> bool:
        """Check if the frame contains at least one detectable face."""
        try:
            app = getattr(self._face_runtime, "_app", None)
            if app is None:
                return True
            faces = app.get(frame)
            return len(faces) > 0
        except Exception:
            return False
