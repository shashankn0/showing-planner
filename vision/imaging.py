"""Loading photos safely, and cheap quality / duplicate checks that need only Pillow + numpy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from vision.config import VisionConfig


def load_rgb(path: Path | str, max_side: int | None = None) -> Image.Image:
    """Open a photo, apply its EXIF rotation, convert to RGB, optionally shrink."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return im.copy()


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash as 16 hex chars. Near-identical photos differ by only a few bits."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@dataclass
class Quality:
    width: int
    height: int
    sharpness: float  # variance of the Laplacian on a 256px copy
    brightness: float  # mean gray level 0..255
    contrast: float  # std of gray level


def quality_metrics(img: Image.Image) -> Quality:
    w, h = img.size
    g = img.convert("L")
    g.thumbnail((256, 256), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    if min(a.shape) < 3:
        return Quality(w, h, 0.0, float(a.mean()), float(a.std()))
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return Quality(w, h, float(lap.var()), float(a.mean()), float(a.std()))


def poor_quality_reason(q: Quality, cfg: VisionConfig) -> str | None:
    """Why a photo is a poor basis for judging condition, or None. Heuristic thresholds."""
    if min(q.width, q.height) < cfg.min_side_px:
        return "low resolution"
    if q.sharpness < cfg.blur_variance_min:
        return "blurry"
    lo, hi = cfg.brightness_range
    if q.brightness < lo:
        return "too dark"
    if q.brightness > hi:
        return "overexposed"
    return None
