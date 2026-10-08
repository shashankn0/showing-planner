"""Constants, the labeling rubric vocabulary, and tunable vision settings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

ROOMS = ["kitchen", "bath", "living", "bedroom", "exterior", "other"]
ROOM_LABELS = {
    "kitchen": "Kitchen",
    "bath": "Bathroom",
    "living": "Living / family",
    "bedroom": "Bedroom",
    "exterior": "Exterior",
    "other": "Other (dining, garage, basement...)",
}
FLAGS = ["water_stain", "damaged_flooring", "dated_finishes", "clutter", "poor_photo_quality"]
FLAG_LABELS = {
    "water_stain": "Water stain",
    "damaged_flooring": "Damaged flooring",
    "dated_finishes": "Dated finishes",
    "clutter": "Clutter",
    "poor_photo_quality": "Poor photo quality",
}
# Photos that are not of the property. They are excluded from scoring, never described.
JUNK_LABELS = {
    "floor_plan": "floor plan",
    "map": "map",
    "logo": "logo or graphic",
    "aerial": "aerial view",
    "document": "document or text page",
    "portrait": "not a property photo",
}
RUBRIC_VERSION = "v1"
CONDITION_LEVELS = (1, 2, 3, 4, 5)
DEFAULT_ROOM_WEIGHTS = {
    "kitchen": 3.0,
    "bath": 2.5,
    "exterior": 2.0,
    "living": 1.5,
    "bedroom": 1.0,
    "other": 0.5,
}
CORE_ROOMS = ("kitchen", "bath", "exterior")  # a listing without these has limited coverage

# Exclusion-reason conventions stored in Photo.exclude_reason
REASON_MANUAL = "manual"  # user excluded it
REASON_KEEP = "manual_keep"  # user overrode an automatic exclusion; never auto-exclude again
REASON_AUTO_PREFIX = "auto: "


@dataclass
class VisionConfig:
    backbone: str = "openclip:ViT-B-32:laion2b_s34b_b79k"
    batch_size: int = 16
    max_side: int = 512
    junk_threshold: float = 0.5
    dedupe_hamming: int = 4
    min_side_px: int = 300
    blur_variance_min: float = 30.0
    brightness_range: tuple[float, float] = (35.0, 235.0)
    min_labels_to_train: int = 40
    holdout_fraction_denominator: int = 5
    extras: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> VisionConfig:
        path = path or CONFIG_DIR / "vision.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        known = {f.name for f in fields(cls)} - {"extras"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "brightness_range" in kwargs:
            kwargs["brightness_range"] = tuple(float(x) for x in kwargs["brightness_range"])
        return cls(**kwargs)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def is_holdout_sha(sha256: str, denominator: int = 5) -> bool:
    """Deterministic holdout assignment: the same photo is always in (or out of) the holdout."""
    return int(sha256[:8], 16) % denominator == 0


def load_room_weights(data_dir: Path | None = None) -> dict[str, float]:
    """Per-user override in <data_dir>/room_weights.yaml, else config/room_weights.yaml."""
    weights = dict(DEFAULT_ROOM_WEIGHTS)
    for path in (
        CONFIG_DIR / "room_weights.yaml",
        (data_dir / "room_weights.yaml") if data_dir else None,
    ):
        if path and path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for room in ROOMS:
                if room in raw:
                    weights[room] = max(0.0, float(raw[room]))
    return weights


def save_room_weights(data_dir: Path, weights: dict[str, float]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    clean = {r: max(0.0, float(weights.get(r, DEFAULT_ROOM_WEIGHTS[r]))) for r in ROOMS}
    (data_dir / "room_weights.yaml").write_text(yaml.safe_dump(clean), encoding="utf-8")
