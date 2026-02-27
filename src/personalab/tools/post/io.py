"""Image I/O and Instagram realism config."""

import cv2
import numpy as np
from pydantic import BaseModel


class IGRealismConfig(BaseModel):
    """Parameters for lightweight Instagram-like degradation."""

    enabled: bool = False
    jpeg_quality: int = 90
    noise_sigma: float = 0.0
    wb_shift: tuple[float, float, float] = (1.0, 1.0, 1.0)
    highlight_clip_p: float = 99.8
    sharpen_amount: float = 0.0
    chroma_aberration_px: float = 0.0
    downscale_max_side: int | None = None


def load_realism_config(cfg: dict) -> IGRealismConfig:
    """Extract Instagram realism config from a global config dict."""
    r = cfg.get("instagram_realism", {})
    return IGRealismConfig(
        enabled=r.get("enabled", False),
        jpeg_quality=r.get("jpeg_quality", 90),
        noise_sigma=r.get("noise_sigma", 0.0),
        wb_shift=tuple(r.get("wb_shift", [1.0, 1.0, 1.0])),
        highlight_clip_p=r.get("highlight_clip_p", 99.8),
        sharpen_amount=r.get("sharpen_amount", 0.0),
        chroma_aberration_px=r.get("chromatic_aberration_px", r.get("chroma_aberration_px", 0.0)),
        downscale_max_side=r.get("downscale_max_side"),
    )


def jpeg_roundtrip(bgr: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG encode/decode to simulate IG compression artifacts."""
    quality = int(np.clip(quality, 30, 100))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else bgr


def add_sensor_noise(
    bgr: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add subtle Gaussian noise to mimic phone sensor + compression grain."""
    if sigma <= 0:
        return bgr
    gen = rng if rng is not None else np.random.default_rng()
    noise = gen.normal(0.0, sigma, bgr.shape).astype(np.float32)
    out = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def wb_shift(bgr: np.ndarray, rgb_mult: tuple[float, float, float]) -> np.ndarray:
    """Apply mild white-balance imperfection (RGB multipliers)."""
    r, g, b = rgb_mult
    mult = np.array([b, g, r], dtype=np.float32).reshape(1, 1, 3)
    out = np.clip(bgr.astype(np.float32) * mult, 0, 255).astype(np.uint8)
    return out


def highlight_clip(bgr: np.ndarray, p: float) -> np.ndarray:
    """Clip highlights slightly to mimic phone HDR + overexposure."""
    p = float(np.clip(p, 95.0, 100.0))
    x = bgr.astype(np.float32)
    thr = np.percentile(x, p)
    if thr <= 0:
        return bgr
    x = np.clip(x, 0, thr)
    x = x * (255.0 / thr)
    return np.clip(x, 0, 255).astype(np.uint8)


def unsharp(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Negative amount softens; positive sharpens slightly."""
    if abs(amount) < 1e-6:
        return bgr
    x = bgr.astype(np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), 1.0)
    out = x + amount * (x - blur)
    return np.clip(out, 0, 255).astype(np.uint8)


def chromatic_aberration(bgr: np.ndarray, px: float) -> np.ndarray:
    """Sub-pixel chromatic aberration via channel shifts."""
    if px <= 0:
        return bgr
    h, w = bgr.shape[:2]
    shift = float(np.clip(px, 0.0, 2.0))

    def shift_chan(chan: np.ndarray, dx: float, dy: float) -> np.ndarray:
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return cv2.warpAffine(chan, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    b, g, r = cv2.split(bgr)
    r2 = shift_chan(r, +shift, 0)
    b2 = shift_chan(b, -shift, 0)
    return cv2.merge([b2, g, r2])


def downscale_if_needed(bgr: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale to a phone-like max resolution to reduce 'too perfect' detail."""
    if not max_side:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    out = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out


def bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    """Decode encoded image bytes (jpg/png/webp) into BGR uint8."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed: bytes are not a valid encoded image.")
    return bgr


def encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    """Encode BGR ndarray to JPEG bytes."""
    quality = int(np.clip(quality, 30, 100))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("cv2.imencode failed.")
    return enc.tobytes()
