"""Tests for Instagram realism tool."""

import numpy as np
import pytest
from personalab import apply_instagram_realism, load_realism_config, IGRealismConfig


def test_load_realism_config_defaults():
    cfg = load_realism_config({})
    assert cfg.enabled is False
    assert cfg.jpeg_quality == 90


def test_load_realism_config_from_dict():
    cfg = load_realism_config({
        "instagram_realism": {
            "enabled": True,
            "jpeg_quality": 85,
            "noise_sigma": 1.0,
        }
    })
    assert cfg.enabled is True
    assert cfg.jpeg_quality == 85
    assert cfg.noise_sigma == 1.0


def test_apply_realism_disabled_returns_decoded_bgr():
    # Minimal 2x2 JPEG-like input (real JPEG bytes)
    from personalab.tools.post.io import bytes_to_bgr, encode_jpeg
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:] = [128, 128, 128]
    jpeg_bytes = encode_jpeg(arr, 90)
    cfg = IGRealismConfig(enabled=False)
    out = apply_instagram_realism(jpeg_bytes, cfg)
    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8


def test_apply_realism_enabled_returns_ndarray():
    from personalab.tools.post.io import encode_jpeg
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[:] = [100, 100, 100]
    jpeg_bytes = encode_jpeg(arr, 95)
    cfg = IGRealismConfig(enabled=True, noise_sigma=0.0, sharpen_amount=0.0)
    out = apply_instagram_realism(jpeg_bytes, cfg, seed=42)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8
