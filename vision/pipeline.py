"""Score listing photos end to end and keep the database up to date.

Per photo: load (EXIF-rotated) -> quality check + near-duplicate hash -> embedding (cached) ->
prediction (room type, condition + confidence, flags) -> automatic exclusion of photos that
aren't of the property (floor plans, maps, logos, headshots) and of near-duplicates.
Per listing: roll the usable photo scores up into one condition score with a worst-room marker.

Re-running is safe: automatic exclusions are recomputed from scratch each time, and anything the
user decided by hand ("manual", "manual_keep") is never overridden.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.config import Settings
from core.db import session_scope
from core.models import ListingScore, Photo, PhotoScore
from sources.photos import resolve_photo_path
from vision.config import (
    JUNK_LABELS,
    REASON_AUTO_PREFIX,
    REASON_KEEP,
    REASON_MANUAL,
    load_room_weights,
)
from vision.embedders import embed_missing
from vision.imaging import dhash, hamming_hex, load_rgb, poor_quality_reason, quality_metrics
from vision.rollup import PhotoRoll, rollup
from vision.runtime import Runtime

ProgressFn = Callable[[str, int, int], None]
MIN_CONTRAST_FOR_HASH = 4.0


@dataclass
class ScoreReport:
    model_version: str
    scored: int = 0
    already_scored: int = 0
    excluded_junk: int = 0
    excluded_duplicates: int = 0
    missing_files: int = 0
    listings_updated: int = 0
    notes: list[str] = field(default_factory=list)


def _clear_auto_exclusions(photos: list[Photo]) -> None:
    for p in photos:
        if p.excluded and (p.exclude_reason or "").startswith(REASON_AUTO_PREFIX):
            p.excluded, p.exclude_reason = False, None


def _cluster_duplicates(photos: list[Photo], max_bits: int) -> list[tuple[Photo, Photo]]:
    """(duplicate, kept) pairs among a listing's photos; keeps the highest-resolution one."""
    pool = [p for p in photos if p.phash and not p.excluded]
    pool.sort(key=lambda p: (-(p.width or 0) * (p.height or 0), p.order_idx, p.id))
    kept: list[Photo] = []
    dupes = []
    for p in pool:
        match = next((k for k in kept if hamming_hex(p.phash, k.phash) <= max_bits), None)
        if match is None:
            kept.append(p)
        elif p.exclude_reason != REASON_KEEP:
            dupes.append((p, match))
    return dupes


def score_photos(
    settings: Settings,
    runtime: Runtime,
    listing_ids: list[int] | None = None,
    rescore: bool = False,
    progress: ProgressFn | None = None,
) -> ScoreReport:
    cfg, predictor = runtime.cfg, runtime.predictor
    report = ScoreReport(model_version=predictor.version)
    weights = load_room_weights(settings.data_dir)

    with session_scope(settings) as s:
        q = select(Photo).order_by(Photo.listing_id, Photo.order_idx, Photo.id)
        if listing_ids is not None:
            q = q.where(Photo.listing_id.in_(listing_ids or [0]))
        photos = [p for p in s.scalars(q) if p.exclude_reason != REASON_MANUAL]
        if not photos:
            return report

        scored_ids = set(
            s.scalars(
                select(PhotoScore.photo_id).where(PhotoScore.model_version == predictor.version)
            )
        )
        todo = [p for p in photos if rescore or p.id not in scored_ids]
        report.already_scored = len(photos) - len(todo)
        _clear_auto_exclusions(photos)

        # ---- 1. quality + hash + embeddings for photos that need scoring
        paths = {p.id: resolve_photo_path(settings, p.path) for p in todo}
        present = [p for p in todo if paths[p.id].exists()]
        report.missing_files = len(todo) - len(present)
        quality_reason: dict[int, str | None] = {}
        for p in present:
            try:
                img = load_rgb(paths[p.id], cfg.max_side)
            except (OSError, ValueError):
                report.missing_files += 1
                continue
            q = quality_metrics(img)
            # A near-flat image has no structure to compare, so every flat image would hash the
            # same and be called a duplicate of the others. Leave those out of duplicate checks.
            p.phash = dhash(img) if q.contrast >= MIN_CONTRAST_FOR_HASH else None
            quality_reason[p.id] = poor_quality_reason(q, cfg)
        present = [p for p in present if p.id in quality_reason]

        def emb_progress(done: int, total: int) -> None:
            if progress:
                progress("Reading photos", done, total)

        vectors, failed = embed_missing(
            runtime.store,
            runtime.embedder,
            [(p.sha256, paths[p.id]) for p in present],
            cfg,
            emb_progress,
        )
        failed_set = set(failed)
        present = [p for p in present if p.sha256 in vectors and p.sha256 not in failed_set]

        # ---- 2. predictions
        if present:
            preds = []
            step = max(cfg.batch_size * 4, 1)
            for i in range(0, len(present), step):
                chunk = present[i : i + step]
                preds += predictor.predict(np.stack([vectors[p.sha256] for p in chunk]))
                if progress:
                    progress("Scoring", min(i + step, len(present)), len(present))
            ids = [p.id for p in present]
            s.execute(
                delete(PhotoScore).where(
                    PhotoScore.photo_id.in_(ids), PhotoScore.model_version == predictor.version
                )
            )
            for p, pred in zip(present, preds, strict=True):
                flags = dict(pred.flags)
                flags["poor_photo_quality"] = max(
                    flags.get("poor_photo_quality", 0.0), 1.0 if quality_reason[p.id] else 0.0
                )
                s.add(
                    PhotoScore(
                        photo_id=p.id,
                        model_version=predictor.version,
                        room_type=pred.room,
                        room_type_conf=pred.room_conf,
                        condition_score=pred.condition,
                        condition_conf=pred.condition_conf,
                        flags={
                            **{k: round(v, 4) for k, v in flags.items()},
                            "_junk": pred.junk if pred.junk_conf >= cfg.junk_threshold else None,
                            "_junk_conf": round(pred.junk_conf, 4),
                            "_quality": quality_reason[p.id],
                        },
                    )
                )
            report.scored = len(present)
        s.flush()

        # ---- 3. automatic exclusions (rebuilt every run) for everything in scope
        latest = {
            sc.photo_id: sc
            for sc in s.scalars(
                select(PhotoScore).where(
                    PhotoScore.photo_id.in_([p.id for p in photos]),
                    PhotoScore.model_version == predictor.version,
                )
            )
        }
        for p in photos:
            sc = latest.get(p.id)
            junk = (sc.flags or {}).get("_junk") if sc else None
            if junk and p.exclude_reason != REASON_KEEP:
                p.excluded, p.exclude_reason = (
                    True,
                    REASON_AUTO_PREFIX + JUNK_LABELS.get(junk, junk),
                )
                report.excluded_junk += 1
        by_listing: dict[int, list[Photo]] = {}
        for p in photos:
            by_listing.setdefault(p.listing_id, []).append(p)
        for group in by_listing.values():
            for dup, kept in _cluster_duplicates(group, cfg.dedupe_hamming):
                dup.excluded, dup.exclude_reason = (
                    True,
                    f"{REASON_AUTO_PREFIX}duplicate of photo #{kept.id}",
                )
                report.excluded_duplicates += 1

        # ---- 4. listing rollups
        for listing_id, group in by_listing.items():
            usable = [
                PhotoRoll(latest[p.id].room_type or "other", latest[p.id].condition_score,
                          latest[p.id].condition_conf or 0.0)
                for p in group
                if not p.excluded and p.id in latest and latest[p.id].condition_score is not None
            ]  # fmt: skip
            result = rollup(usable, weights)
            s.execute(
                delete(ListingScore).where(
                    ListingScore.listing_id == listing_id,
                    ListingScore.model_version == predictor.version,
                )
            )
            s.add(
                ListingScore(
                    listing_id=listing_id,
                    model_version=predictor.version,
                    overall=result.overall,
                    worst_room=result.worst_room,
                    worst_room_score=result.worst_room_score,
                    by_room=result.as_json(),
                    weights=weights,
                    n_photos_used=result.n_photos_used,
                )
            )
            report.listings_updated += 1
    return report


def refresh_rollups(
    settings: Settings, model_version: str, listing_ids: list[int] | None = None
) -> int:
    """Recompute listing scores from stored photo scores (e.g. after editing room weights)."""
    weights = load_room_weights(settings.data_dir)
    n = 0
    with session_scope(settings) as s:
        q = select(Photo)
        if listing_ids is not None:
            q = q.where(Photo.listing_id.in_(listing_ids or [0]))
        by_listing: dict[int, list[Photo]] = {}
        for p in s.scalars(q):
            by_listing.setdefault(p.listing_id, []).append(p)
        scores = {
            sc.photo_id: sc
            for sc in s.scalars(select(PhotoScore).where(PhotoScore.model_version == model_version))
        }
        for listing_id, group in by_listing.items():
            usable = [
                PhotoRoll(scores[p.id].room_type or "other", scores[p.id].condition_score,
                          scores[p.id].condition_conf or 0.0)
                for p in group
                if not p.excluded and p.id in scores and scores[p.id].condition_score is not None
            ]  # fmt: skip
            if not usable and not any(p.id in scores for p in group):
                continue
            result = rollup(usable, weights)
            s.execute(
                delete(ListingScore).where(
                    ListingScore.listing_id == listing_id,
                    ListingScore.model_version == model_version,
                )
            )
            s.add(
                ListingScore(
                    listing_id=listing_id, model_version=model_version, overall=result.overall,
                    worst_room=result.worst_room, worst_room_score=result.worst_room_score,
                    by_room=result.as_json(), weights=weights, n_photos_used=result.n_photos_used,
                )
            )  # fmt: skip
            n += 1
    return n


def photo_scores_for(
    session: Session, photo_ids: list[int], model_version: str
) -> dict[int, PhotoScore]:
    return {
        sc.photo_id: sc
        for sc in session.scalars(
            select(PhotoScore).where(
                PhotoScore.photo_id.in_(photo_ids or [0]), PhotoScore.model_version == model_version
            )
        )
    }
