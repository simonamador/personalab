"""Tests for FrameExtractor using synthetic video data."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from personalab.identity.frame_extractor import FrameExtractor, ExtractedFrame


def _make_synthetic_video(n_frames: int = 30, fps: float = 30.0, width: int = 64, height: int = 64) -> bytes:
    """Create a minimal MP4 video in memory and return the bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), fill_value=(i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


class TestFrameExtractorUniform:
    def test_extracts_correct_number_of_frames(self):
        video_data = _make_synthetic_video(n_frames=30)
        extractor = FrameExtractor()
        frames = extractor.extract(video_data, strategy="uniform", max_frames=5)
        assert len(frames) == 5

    def test_all_frames_are_extracted_frame_type(self):
        video_data = _make_synthetic_video(n_frames=10)
        extractor = FrameExtractor()
        frames = extractor.extract(video_data, strategy="uniform", max_frames=3)
        for f in frames:
            assert isinstance(f, ExtractedFrame)
            assert isinstance(f.image, np.ndarray)
            assert f.image.shape == (64, 64, 3)
            assert f.frame_index >= 0
            assert f.timestamp_s >= 0.0

    def test_max_frames_capped_by_total(self):
        video_data = _make_synthetic_video(n_frames=3)
        extractor = FrameExtractor()
        frames = extractor.extract(video_data, strategy="uniform", max_frames=10)
        assert len(frames) <= 3

    def test_empty_video_returns_empty(self):
        extractor = FrameExtractor()
        frames = extractor.extract(b"", strategy="uniform", max_frames=5)
        assert frames == []

    def test_frame_indices_are_evenly_spaced(self):
        video_data = _make_synthetic_video(n_frames=100)
        extractor = FrameExtractor()
        frames = extractor.extract(video_data, strategy="uniform", max_frames=5)
        indices = [f.frame_index for f in frames]
        assert indices[0] == 0
        assert indices[-1] == 99
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        assert all(d > 0 for d in diffs)

    def test_timestamps_increase(self):
        video_data = _make_synthetic_video(n_frames=30, fps=15.0)
        extractor = FrameExtractor()
        frames = extractor.extract(video_data, strategy="uniform", max_frames=4)
        timestamps = [f.timestamp_s for f in frames]
        assert timestamps == sorted(timestamps)


class TestFrameExtractorFaceDetected:
    def test_falls_back_to_uniform_without_face_runtime(self):
        video_data = _make_synthetic_video(n_frames=20)
        extractor = FrameExtractor(face_runtime=None)
        frames = extractor.extract(video_data, strategy="face_detected", max_frames=4)
        assert len(frames) > 0
