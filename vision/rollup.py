"""Combine per-photo scores into one condition score per listing.

Rules (all visible in the output):
  * Only usable photos count: not excluded, and with a condition score.
  * Within a room type, photos are averaged, trusting confident predictions more.
  * Across room types, rooms are weighted by importance (config/room_weights.yaml); room types
    with no photos are skipped and the rest re-weighted.
  * "Worst room" is the lowest-scoring room type among those that carry weight, so one bad
    kitchen can't hide behind five nice bedrooms. ("other" rooms are never called the worst.)
  * Coverage is reported: a listing with no kitchen, bath or exterior photo is flagged limited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vision.config import CORE_ROOMS, ROOMS

MIN_PHOTO_WEIGHT = 0.2  # even a low-confidence photo counts a little


@dataclass
class PhotoRoll:
    room: str
    score: float  # 1..5
    confidence: float  # 0..1


@dataclass
class RollupResult:
    overall: float | None
    worst_room: str | None
    worst_room_score: float | None
    by_room: dict[str, dict[str, float]]  # room -> {score, n, confidence}
    missing_core_rooms: list[str] = field(default_factory=list)
    n_photos_used: int = 0

    @property
    def limited_coverage(self) -> bool:
        return bool(self.missing_core_rooms) or self.n_photos_used < 5

    def as_json(self) -> dict:
        return {
            "rooms": self.by_room,
            "missing_core_rooms": self.missing_core_rooms,
            "limited_coverage": self.limited_coverage,
        }


def rollup(photos: list[PhotoRoll], room_weights: dict[str, float]) -> RollupResult:
    by_room: dict[str, dict[str, float]] = {}
    for room in ROOMS:
        items = [p for p in photos if p.room == room]
        if not items:
            continue
        w = [max(p.confidence, MIN_PHOTO_WEIGHT) for p in items]
        score = sum(p.score * wi for p, wi in zip(items, w, strict=True)) / sum(w)
        by_room[room] = {
            "score": round(score, 3),
            "n": float(len(items)),
            "confidence": round(sum(p.confidence for p in items) / len(items), 3),
        }
    missing = [r for r in CORE_ROOMS if r not in by_room]
    weighted = [(by_room[r]["score"], room_weights.get(r, 0.0)) for r in by_room]
    total_w = sum(w for _, w in weighted)
    if not by_room or total_w <= 0:
        return RollupResult(None, None, None, by_room, missing, len(photos))
    overall = sum(s * w for s, w in weighted) / total_w
    candidates = {
        r: v["score"] for r, v in by_room.items() if r != "other" and room_weights.get(r, 0) > 0
    }
    worst = min(candidates, key=candidates.get) if candidates else None
    return RollupResult(
        overall=round(overall, 3),
        worst_room=worst,
        worst_room_score=candidates[worst] if worst else None,
        by_room=by_room,
        missing_core_rooms=missing,
        n_photos_used=len(photos),
    )
