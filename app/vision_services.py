"""Photo-scoring actions for the UI (kept separate from app.services so the core app imports
cleanly when the optional vision packages are not installed)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select

from core.config import Settings
from core.db import session_scope
from core.models import Label, Listing, ListingScore, Photo, PhotoScore
from sources.photos import resolve_photo_path
from vision import registry
from vision.active import (
    COLD_START_LABELS,
    HOLDOUT_EVERY,
    interleave_holdout,
    pick_diverse,
    pick_informative,
)
from vision.config import (
    FLAGS,
    REASON_AUTO_PREFIX,
    REASON_KEEP,
    REASON_MANUAL,
    ROOMS,
    RUBRIC_VERSION,
    VisionConfig,
    is_holdout_sha,
)
from vision.embedders import EmbeddingStore, VisionUnavailable, deps_available, embed_missing
from vision.pipeline import ScoreReport, refresh_rollups, score_photos
from vision.runtime import Runtime, build_runtime, required_backbone
from vision.train import TrainingError, collect_labeled, train_and_evaluate

__all__ = ["TrainingError"]


def vision_ready() -> bool:
    """True when the optional packages needed for CLIP-based scoring are installed."""
    d = deps_available()
    return d["torch"] and d["open_clip"] and d["scikit-learn"]


def active_model_version(settings: Settings) -> str | None:
    return registry.active_version(settings.models_dir)


def make_runtime(settings: Settings) -> Runtime:
    return build_runtime(settings, VisionConfig.load())


# ------------------------------------------------------------------ scoring


def score_listings(
    settings: Settings,
    runtime: Runtime,
    listing_ids: list[int] | None = None,
    rescore: bool = False,
    progress=None,  # noqa: ANN001
) -> ScoreReport:
    return score_photos(settings, runtime, listing_ids, rescore, progress)


@dataclass
class ReviewPhoto:
    photo: Photo
    room: str | None = None
    room_conf: float | None = None
    condition: float | None = None
    condition_conf: float | None = None
    flags: dict[str, float] = field(default_factory=dict)
    quality_note: str | None = None


@dataclass
class ListingReview:
    listing: Listing
    version: str | None
    overall: float | None
    worst_room: str | None
    worst_room_score: float | None
    by_room: dict
    limited_coverage: bool
    missing_core_rooms: list[str]
    photos: list[ReviewPhoto]


def _score_version(settings: Settings, session, listing_id: int) -> str | None:  # noqa: ANN001
    """Prefer the active model's scores; otherwise the most recent."""
    active = active_model_version(settings)
    q = select(ListingScore).where(ListingScore.listing_id == listing_id)
    rows = list(session.scalars(q.order_by(ListingScore.created_at)))
    if not rows:
        return None
    for r in rows:
        if active and r.model_version == active:
            return active
    return rows[-1].model_version


def review_listing(settings: Settings, listing_id: int) -> ListingReview | None:
    with session_scope(settings) as s:
        listing = s.get(Listing, listing_id)
        if listing is None:
            return None
        version = _score_version(settings, s, listing_id)
        ls = (
            s.scalar(
                select(ListingScore).where(
                    ListingScore.listing_id == listing_id, ListingScore.model_version == version
                )
            )
            if version
            else None
        )
        photos = list(listing.photos)
        scores = (
            {
                sc.photo_id: sc
                for sc in s.scalars(
                    select(PhotoScore).where(
                        PhotoScore.photo_id.in_([p.id for p in photos] or [0]),
                        PhotoScore.model_version == version,
                    )
                )
            }
            if version
            else {}
        )
        items = []
        for p in photos:
            sc = scores.get(p.id)
            flags = (
                {k: v for k, v in (sc.flags or {}).items() if not k.startswith("_")} if sc else {}
            )
            items.append(
                ReviewPhoto(
                    photo=p,
                    room=sc.room_type if sc else None,
                    room_conf=sc.room_type_conf if sc else None,
                    condition=sc.condition_score if sc else None,
                    condition_conf=sc.condition_conf if sc else None,
                    flags=flags,
                    quality_note=(sc.flags or {}).get("_quality") if sc else None,
                )
            )
        by_room = (ls.by_room or {}) if ls else {}
        return ListingReview(
            listing=listing,
            version=version,
            overall=ls.overall if ls else None,
            worst_room=ls.worst_room if ls else None,
            worst_room_score=ls.worst_room_score if ls else None,
            by_room=by_room.get("rooms", {}),
            limited_coverage=bool(by_room.get("limited_coverage")),
            missing_core_rooms=list(by_room.get("missing_core_rooms", [])),
            photos=items,
        )


def set_photo_excluded(settings: Settings, photo_id: int, excluded: bool) -> None:
    """User decision from the review grid. Overrides automatic exclusion in either direction."""
    with session_scope(settings) as s:
        p = s.get(Photo, photo_id)
        if p is None:
            return
        was_auto = (p.exclude_reason or "").startswith(REASON_AUTO_PREFIX)
        if excluded:
            p.excluded, p.exclude_reason = True, REASON_MANUAL
        else:
            p.excluded, p.exclude_reason = False, REASON_KEEP if was_auto else None
        listing_id = p.listing_id
    version = active_model_version(settings) or _any_version(settings, listing_id)
    if version:
        refresh_rollups(settings, version, [listing_id])


def _any_version(settings: Settings, listing_id: int) -> str | None:
    with session_scope(settings) as s:
        return _score_version(settings, s, listing_id)


# ------------------------------------------------------------------ labeling


@dataclass
class LabelStats:
    labeled: int  # photos with a condition label
    unsure: int
    holdout_labeled: int
    available: int  # unlabeled, non-excluded photos
    embedded: int  # of those, ones with a cached embedding


def _store(settings: Settings, cfg: VisionConfig | None = None) -> EmbeddingStore:
    return EmbeddingStore(settings.embeddings_dir, required_backbone(settings, cfg))


def label_stats(settings: Settings) -> LabelStats:
    cfg = VisionConfig.load()
    store = _store(settings, cfg)
    with session_scope(settings) as s:
        latest: dict[int, Label] = {}
        for lab in s.scalars(select(Label).order_by(Label.id)):
            latest[lab.photo_id] = lab
        scored = {pid: lab for pid, lab in latest.items() if not lab.unsure and lab.condition}
        shas = dict(s.execute(select(Photo.id, Photo.sha256)).all())
        free = [
            p
            for p in s.scalars(select(Photo).where(Photo.excluded.is_(False)))
            if p.id not in latest
        ]
        return LabelStats(
            labeled=len(scored),
            unsure=sum(1 for lab in latest.values() if lab.unsure),
            holdout_labeled=sum(
                is_holdout_sha(shas[pid], cfg.holdout_fraction_denominator)
                for pid in scored
                if pid in shas
            ),
            available=len(free),
            embedded=sum(store.get(p.sha256) is not None for p in free),
        )


def prepare_embeddings(settings: Settings, runtime: Runtime, progress=None) -> int:  # noqa: ANN001
    """Read every usable photo once so it can be ordered for labeling. Returns photos embedded."""
    with session_scope(settings) as s:
        items = [
            (p.sha256, resolve_photo_path(settings, p.path))
            for p in s.scalars(select(Photo).where(Photo.excluded.is_(False)))
        ]
    vecs, _ = embed_missing(runtime.store, runtime.embedder, items, runtime.cfg, progress)
    return len(vecs)


@dataclass
class QueueItem:
    photo_id: int
    strategy: str  # shown to the user


class _NoEmbedder:
    """Stand-in used when only cached embeddings may be read (never loads a model)."""

    id = "cache-only"
    supports_text = False

    def embed_images(self, images):  # noqa: ANN001, ANN201
        raise VisionUnavailable("Photos need to be prepared first.")

    def embed_texts(self, texts):  # noqa: ANN001, ANN201
        raise VisionUnavailable("not available")


def labeling_queue(
    settings: Settings,
    k: int = 20,
    seed: int | None = None,
    zeroshot=None,  # noqa: ANN001  (optional ZeroShotModel to guess room types)
) -> list[QueueItem]:
    """The next photos to label: diverse at first, then most informative, with holdout checks mixed in.

    Works from cached embeddings only, so opening the labeling page never loads a model.
    """
    cfg = VisionConfig.load()
    store = _store(settings, cfg)
    rng = random.Random(seed)
    with session_scope(settings) as s:
        labeled_ids = set(s.scalars(select(Label.photo_id)))
        photos = [
            p
            for p in s.scalars(select(Photo).where(Photo.excluded.is_(False)).order_by(Photo.id))
            if p.id not in labeled_ids
        ]
        known_rooms = {
            sc.photo_id: sc.room_type
            for sc in s.scalars(select(PhotoScore).order_by(PhotoScore.id))
            if sc.room_type
        }
    if not photos:
        return []
    hold_ids = {p.id for p in photos if is_holdout_sha(p.sha256, cfg.holdout_fraction_denominator)}
    holdout = [p for p in photos if p.id in hold_ids]
    train = [p for p in photos if p.id not in hold_ids]
    n_hold = max(1, k // HOLDOUT_EVERY) if holdout else 0
    hold_pick = rng.sample(holdout, min(n_hold, len(holdout)))

    have = [(p, store.get(p.sha256)) for p in train]
    have = [(p, v) for p, v in have if v is not None]
    if not have:  # nothing prepared yet: random order
        order = rng.sample(train, min(k, len(train)))
        strategy = "Random order (photos not prepared yet)"
        train_items = [(p.id, strategy) for p in order]
    else:
        tp = [p for p, _ in have]
        tX = np.stack([v for _, v in have])
        rooms = [known_rooms.get(p.id, "other") for p in tp]
        if zeroshot is not None:
            rooms = [o.room for o in zeroshot.predict(tX)]
        n_lab = label_stats(settings).labeled
        idx, strategy = None, "Diverse sample"
        if n_lab >= COLD_START_LABELS:
            try:
                with session_scope(settings) as s:
                    ls = collect_labeled(settings, s, store, _NoEmbedder(), cfg)
                ok = ls.scored
                if ok.sum() >= 20:
                    idx = pick_informative(
                        tX, rooms, ls.X[ok],
                        [r for r, keep in zip(ls.rooms, ok, strict=True) if keep], ls.y[ok], k,
                        seed=seed or 0,
                    )  # fmt: skip
                    strategy = "Most informative"
            except VisionUnavailable:
                idx = None
        if idx is None:
            idx = pick_diverse(tX, rooms, k, seed=seed or 0)
        train_items = [(tp[i].id, strategy) for i in idx]

    seen: set[int] = set()
    queue: list[QueueItem] = []
    for kind, i in interleave_holdout(list(range(len(train_items))), list(range(len(hold_pick)))):
        pid, strat = (
            train_items[i] if kind == "train" else (hold_pick[i].id, "Random check (holdout)")
        )
        if pid not in seen:
            seen.add(pid)
            queue.append(QueueItem(pid, strat))
    return queue


def save_label(
    settings: Settings,
    photo_id: int,
    *,
    condition: int | None,
    room: str,
    flags: list[str],
    unsure: bool = False,
    ms_spent: int | None = None,
    holdout_denominator: int = 5,
) -> int:
    if not unsure and condition not in (1, 2, 3, 4, 5):
        raise ValueError("Condition must be 1 to 5.")
    if room not in ROOMS:
        raise ValueError("Pick a room type.")
    with session_scope(settings) as s:
        photo = s.get(Photo, photo_id)
        if photo is None:
            raise ValueError("That photo no longer exists.")
        lab = Label(
            photo_id=photo_id,
            condition=None if unsure else int(condition),
            room_type=room,
            flags=[f for f in flags if f in FLAGS],
            unsure=unsure,
            rubric_version=RUBRIC_VERSION,
            is_holdout=is_holdout_sha(photo.sha256, holdout_denominator),
            ms_spent=ms_spent,
        )
        s.add(lab)
        s.flush()
        return lab.id


def undo_last_label(settings: Settings) -> int | None:
    """Delete the most recent label. Returns the photo id so it can be shown again."""
    with session_scope(settings) as s:
        lab = s.scalar(select(Label).order_by(Label.id.desc()).limit(1))
        if lab is None:
            return None
        pid = lab.photo_id
        s.delete(lab)
        return pid


def exclude_photo(settings: Settings, photo_id: int) -> None:
    with session_scope(settings) as s:
        p = s.get(Photo, photo_id)
        if p is not None:
            p.excluded, p.exclude_reason = True, REASON_MANUAL


def zero_shot_guess(runtime: Runtime | None, settings: Settings, photo_id: int) -> dict | None:
    """What the baseline thinks of one photo (shown only if the user asks)."""
    if runtime is None or runtime.zeroshot is None:
        return None
    with session_scope(settings) as s:
        p = s.get(Photo, photo_id)
        sha = p.sha256 if p else None
    vec = runtime.store.get(sha) if sha else None
    if vec is None:
        return None
    out = runtime.zeroshot.predict(vec[None, :])[0]
    return {"room": out.room, "condition": out.condition, "flags": out.flags}


# ------------------------------------------------------------------ training


def train_model(settings: Settings, runtime: Runtime, progress=None) -> tuple[str, dict]:  # noqa: ANN001
    """Train from labels and save (not activate). Raises TrainingError with a helpful message."""
    with session_scope(settings) as s:
        bundle, metrics, card = train_and_evaluate(
            settings, s, runtime.embedder, runtime.cfg, runtime.zeroshot, progress
        )
    registry.save_version(settings.models_dir, bundle, metrics, card)
    return bundle.version, metrics


def activate_model(settings: Settings, version: str | None) -> None:
    registry.set_active(settings.models_dir, version)


def model_versions(settings: Settings) -> list[dict]:
    return registry.list_versions(settings.models_dir)


def model_card(settings: Settings, version: str) -> str:
    path = settings.models_dir / version / "model_card.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def score_summary(settings: Settings) -> dict:
    """How many photos/listings have scores (for status displays)."""
    with session_scope(settings) as s:
        return {
            "photos": s.scalar(select(func.count()).select_from(Photo)) or 0,
            "scored_photos": s.scalar(select(func.count(func.distinct(PhotoScore.photo_id)))) or 0,
            "scored_listings": s.scalar(select(func.count(func.distinct(ListingScore.listing_id))))
            or 0,
        }


def metrics_json(settings: Settings, version: str) -> dict:
    f = settings.models_dir / version / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


_ = (REASON_KEEP, EmbeddingStore)  # re-exported for callers/tests that import from here
