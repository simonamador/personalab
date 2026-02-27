"""Apply Instagram-like realism pass to image bytes."""

import numpy as np

from personalab.tools.post.io import (
    bytes_to_bgr,
    jpeg_roundtrip,
    downscale_if_needed,
    wb_shift,
    highlight_clip,
    unsharp,
    chromatic_aberration,
    add_sensor_noise,
    IGRealismConfig,
)


def apply_instagram_realism(
    img: bytes,
    cfg: IGRealismConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a deterministic-ish IG realism pass (optionally seeded). Returns BGR ndarray."""
    rng = np.random.default_rng(int(seed) % (2**32 - 1) if seed is not None else None)

    bgr = bytes_to_bgr(img)

    if not cfg.enabled:
        return bgr

    out = bgr.copy()
    out = downscale_if_needed(out, cfg.downscale_max_side)
    out = wb_shift(out, cfg.wb_shift)
    out = highlight_clip(out, cfg.highlight_clip_p)
    out = unsharp(out, cfg.sharpen_amount)
    out = chromatic_aberration(out, cfg.chroma_aberration_px)
    out = add_sensor_noise(out, cfg.noise_sigma, rng=rng)
    out = jpeg_roundtrip(out, cfg.jpeg_quality)
    return out
