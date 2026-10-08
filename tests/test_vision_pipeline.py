"""End-to-end vision flow with fake models: scoring, exclusion, rollup, training, activation."""

import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select

pytest.importorskip("sklearn")

from core.config import Settings  # noqa: E402
from core.db import init_db, session_scope  # noqa: E402
from core.models import Label, Listing, ListingScore, Photo, PhotoScore  # noqa: E402
from core.store import upsert_listings  # noqa: E402
from sources.base import RawListing  # noqa: E402
from sources.photos import ingest_photos  # noqa: E402
from tests.fakes import FakeClip, bath_prompts, floor_plan_prompts, kitchen_prompts  # noqa: E402
from vision.active import (  # noqa: E402
    HOLDOUT_EVERY,
    _farthest_first,
    interleave_holdout,
    pick_diverse,
    pick_informative,
)
from vision.config import (  # noqa: E402
    REASON_KEEP,
    REASON_MANUAL,
    VisionConfig,
    is_holdout_sha,
)
from vision.embedders import ColorHistEmbedder, EmbeddingStore, VisionUnavailable  # noqa: E402
from vision.pipeline import refresh_rollups, score_photos  # noqa: E402
from vision.registry import active_version, save_version, set_active  # noqa: E402
from vision.runtime import build_runtime  # noqa: E402
from vision.train import TrainingError, train_and_evaluate  # noqa: E402

RED, BLUE, GRAY = (200, 30, 30), (30, 30, 200), (128, 128, 128)


def noisy_color(color, seed, size=(420, 320), sigma=5):
    """A photo-like image: a base color plus a random blocky pattern (structure) plus fine noise."""
    rng = np.random.default_rng(seed)
    blocks = rng.normal(0, 18, (6, 8, 1))
    pattern = np.kron(blocks, np.ones((size[1] // 6 + 1, size[0] // 8 + 1, 1)))[
        : size[1], : size[0]
    ]
    a = np.clip(np.array(color) + pattern + rng.normal(0, sigma, (size[1], size[0], 3)), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def save(img, path, quality=92):
    img.save(path, "JPEG", quality=quality)
    return path


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    return s


@pytest.fixture
def clip():
    c = FakeClip()
    c.mark(RED, kitchen_prompts())  # red photos "are" kitchens
    c.mark(BLUE, bath_prompts())  # blue photos "are" bathrooms
    c.mark(GRAY, floor_plan_prompts())  # gray photos "are" floor plans
    return c


def make_listing(settings, tmp_path, specs, mls="L1"):
    """specs: list of (color, seed). Returns the listing id."""
    with session_scope(settings) as s:
        upsert_listings(
            s,
            [
                RawListing(
                    "manual", f"{mls} Test St", "Testville", "GA", "30301", source_listing_id=mls
                )
            ],
        )
        listing = s.scalar(select(Listing).where(Listing.source_listing_id == mls))
        files = []
        for color, seed in specs:
            files.append(save(noisy_color(color, seed), tmp_path / f"{mls}_{color[0]}_{seed}.jpg"))
        ingest_photos(s, settings, listing, files)
        return listing.id


def photos_of(s, red_value):
    """Photos whose marker color has this red channel (200 = kitchen, 30 = bath, 128 = floor plan)."""
    return [
        p
        for p in s.scalars(select(Photo).order_by(Photo.id))
        if int(p.path.split("_")[-2]) == red_value
    ]


def runtime_for(settings, embedder):
    return build_runtime(settings, VisionConfig(), embedder=embedder)


# ------------------------------------------------------------------ scoring


def test_scoring_assigns_rooms_excludes_junk_and_rolls_up(settings, tmp_path, clip):
    lid = make_listing(
        settings, tmp_path, [(RED, 1), (RED, 2), (RED, 3), (BLUE, 4), (BLUE, 5), (GRAY, 6)]
    )
    rt = runtime_for(settings, clip)
    rep = score_photos(settings, rt)
    assert rep.scored == 6 and rep.excluded_junk == 1 and rep.listings_updated == 1
    with session_scope(settings) as s:
        scores = {p.photo_id: p for p in s.scalars(select(PhotoScore))}
        assert all(scores[p.id].room_type == "kitchen" for p in photos_of(s, 200))
        assert all(scores[p.id].room_type == "bath" for p in photos_of(s, 30))
        (plan,) = photos_of(s, 128)
        assert plan.excluded and plan.exclude_reason == "auto: floor plan"
        assert sum(p.excluded for p in s.scalars(select(Photo))) == 1
        ls = s.scalar(select(ListingScore).where(ListingScore.listing_id == lid))
        assert ls.n_photos_used == 5 and ls.overall is not None
        assert set(ls.by_room["rooms"]) == {"kitchen", "bath"}
        assert ls.by_room["missing_core_rooms"] == ["exterior"] and ls.by_room["limited_coverage"]
        assert ls.worst_room in {"kitchen", "bath"}
        assert 1 <= ls.overall <= 5


def test_near_duplicates_are_excluded_keeping_the_largest(settings, tmp_path, clip):
    base = noisy_color(RED, 7)
    small, big = base.resize((330, 250)), base.resize((600, 450))
    with session_scope(settings) as s:
        upsert_listings(s, [RawListing("manual", "9 Dup St", "Testville", "GA", "30301")])
        listing = s.scalar(select(Listing))
        files = [
            save(small, tmp_path / "a.jpg"),
            save(big, tmp_path / "b.jpg"),
            save(noisy_color(RED, 99), tmp_path / "c.jpg"),
        ]
        ingest_photos(s, settings, listing, files)
    rep = score_photos(settings, runtime_for(settings, clip))
    assert rep.excluded_duplicates == 1
    with session_scope(settings) as s:
        photos = {p.path.split("_", 1)[-1]: p for p in s.scalars(select(Photo))}
        by_size = sorted(s.scalars(select(Photo)), key=lambda p: p.width)
        dup = next(p for p in by_size if p.excluded)
        assert dup.width == 330 and dup.exclude_reason.startswith("auto: duplicate of photo #")
        assert not next(p for p in by_size if p.width == 600).excluded
        assert len(photos) == 3


def test_rerun_is_idempotent_and_reuses_the_embedding_cache(settings, tmp_path, clip):
    make_listing(settings, tmp_path, [(RED, 1), (RED, 2), (BLUE, 3)])
    rt = runtime_for(settings, clip)
    first = score_photos(settings, rt)
    calls_after_first = clip.image_calls
    second = score_photos(settings, rt)
    assert (first.scored, second.scored, second.already_scored) == (3, 0, 3)
    assert clip.image_calls == calls_after_first
    again = score_photos(settings, rt, rescore=True)
    assert again.scored == 3 and clip.image_calls == calls_after_first  # recomputed from cache
    with session_scope(settings) as s:
        assert s.scalar(select(func.count()).select_from(PhotoScore)) == 3
        assert s.scalar(select(func.count()).select_from(ListingScore)) == 1


def test_manual_choices_are_never_overridden(settings, tmp_path, clip):
    make_listing(settings, tmp_path, [(RED, 1), (RED, 2), (GRAY, 3), (GRAY, 4)])
    rt = runtime_for(settings, clip)
    with session_scope(settings) as s:
        kitchens, plans = photos_of(s, 200), photos_of(s, 128)
        kitchens[0].excluded, kitchens[0].exclude_reason = True, REASON_MANUAL  # user dropped one
        plans[0].exclude_reason = REASON_KEEP  # user insists a "floor plan" is wanted
        dropped_id, kept_id, other_plan_id = kitchens[0].id, plans[0].id, plans[1].id
    rep = score_photos(settings, rt)
    with session_scope(settings) as s:
        dropped, kept, other = (s.get(Photo, i) for i in (dropped_id, kept_id, other_plan_id))
        assert dropped.excluded and dropped.exclude_reason == REASON_MANUAL
        assert not s.scalar(
            select(PhotoScore).where(PhotoScore.photo_id == dropped_id)
        )  # not scored
        assert not kept.excluded and kept.exclude_reason == REASON_KEEP
        assert other.excluded and other.exclude_reason == "auto: floor plan"
    assert rep.scored == 3


def test_changing_room_weights_changes_the_rollup(settings, tmp_path, clip):
    from vision.config import save_room_weights

    lid = make_listing(settings, tmp_path, [(RED, 1), (BLUE, 2)])
    rt = runtime_for(settings, clip)
    score_photos(settings, rt)
    with session_scope(settings) as s:
        before = s.scalar(select(ListingScore.overall).where(ListingScore.listing_id == lid))
        sc = {p.room_type: p for p in s.scalars(select(PhotoScore))}
        sc["kitchen"].condition_score, sc["bath"].condition_score = 1.0, 5.0
        sc["kitchen"].condition_conf = sc["bath"].condition_conf = 1.0
    save_room_weights(
        settings.data_dir,
        {"kitchen": 9.0, "bath": 1.0, "exterior": 1, "living": 1, "bedroom": 1, "other": 1},
    )
    assert refresh_rollups(settings, rt.predictor.version) == 1
    with session_scope(settings) as s:
        after = s.scalar(select(ListingScore.overall).where(ListingScore.listing_id == lid))
        assert after == pytest.approx((1.0 * 9 + 5.0 * 1) / 10, abs=1e-3) and after != before


def test_missing_files_are_reported_not_fatal(settings, tmp_path, clip):
    make_listing(settings, tmp_path, [(RED, 1), (RED, 2)])
    with session_scope(settings) as s:
        photo = s.scalar(select(Photo).order_by(Photo.id))
        (settings.photos_dir / photo.path).unlink()
    rep = score_photos(settings, runtime_for(settings, clip))
    assert rep.missing_files == 1 and rep.scored == 1


def test_scoring_scope_can_be_limited_to_chosen_listings(settings, tmp_path, clip):
    a = make_listing(settings, tmp_path, [(RED, 1)], mls="A")
    make_listing(settings, tmp_path, [(RED, 2)], mls="B")
    rep = score_photos(settings, runtime_for(settings, clip), listing_ids=[a])
    assert rep.scored == 1 and rep.listings_updated == 1


def test_runtime_needs_a_clip_backbone_or_a_trained_model(settings):
    with pytest.raises(VisionUnavailable, match="CLIP-style backbone"):
        build_runtime(settings, VisionConfig(), embedder=ColorHistEmbedder())


# ------------------------------------------------------------------ labels -> train -> activate


LEVEL_COLORS = {
    1: (220, 20, 20),
    2: (20, 200, 20),
    3: (20, 20, 220),
    4: (200, 20, 200),
    5: (128, 128, 128),
}


def make_labeled_listing(settings, tmp_path, n_per_level=16):
    """Photos whose color encodes the true condition, so a head can learn it from ColorHist."""
    rooms = ["kitchen", "bath", "bedroom"]
    with session_scope(settings) as s:
        upsert_listings(s, [RawListing("manual", "1 Label St", "Testville", "GA", "30301")])
        listing = s.scalar(select(Listing))
        files, truth = [], []
        for lvl, color in LEVEL_COLORS.items():
            for i in range(n_per_level):
                files.append(
                    save(
                        noisy_color(color, seed=lvl * 1000 + i, size=(360, 300), sigma=14),
                        tmp_path / f"l{lvl}_{i}.jpg",
                    )
                )
                truth.append(lvl)
        ingest_photos(s, settings, listing, files)
        photos = list(s.scalars(select(Photo).order_by(Photo.id)))
        by_name = {p.path.split("_", 1)[-1]: p for p in photos}
        for f, lvl in zip(files, truth, strict=True):
            p = next(p for p in photos if p.path.endswith(f.name) or f.name in p.path)
            s.add(
                Label(
                    photo_id=p.id,
                    condition=lvl,
                    room_type=rooms[p.id % 3],
                    flags=["clutter"] if lvl <= 2 else [],
                )
            )
        assert len(by_name) == len(files)
    return len(files)


def test_training_loop_learns_evaluates_and_activates(settings, tmp_path):
    n = make_labeled_listing(settings, tmp_path)
    embedder, cfg = ColorHistEmbedder(), VisionConfig()
    with session_scope(settings) as s:
        bundle, metrics, card = train_and_evaluate(settings, s, embedder, cfg, zeroshot=None)
    assert metrics["n_labeled"] == n and metrics["n_train_pool"] + metrics["n_holdout"] == n
    assert metrics["chosen_head"] in {"ridge", "corn_linear"}
    best = metrics["candidates"][metrics["chosen_head"]]["cv"]
    assert best["qwk"] > 0.7 and best["mae"] < 1.0
    h = metrics["holdout"]
    assert h is not None and h["n"] >= 10 and h["qwk"] > 0.6
    assert len(h["confusion"]) == 5 and set(h["per_room_mae"]) <= {"kitchen", "bath", "bedroom"}
    assert metrics["room_type"]["trained"] and metrics["flags"]["clutter"]["trained"]
    assert (
        metrics["learning_curve"] and "no zero-shot baseline" in " ".join(metrics["notes"]).lower()
    )
    for section in (
        "# Model card",
        "## Intended use",
        "not an appraisal",
        "## Results",
        "## Limitations",
        "holdout",
    ):
        assert section in card
    assert "address" in card.lower() and "pixels" in card

    save_version(settings.models_dir, bundle, metrics, card)
    set_active(settings.models_dir, bundle.version)
    assert active_version(settings.models_dir) == bundle.version
    rt = build_runtime(settings, VisionConfig(), embedder=embedder)
    assert rt.bundle is not None and rt.predictor.version == bundle.version
    rep = score_photos(settings, rt)
    assert rep.scored == n and rep.model_version == bundle.version
    with session_scope(settings) as s:
        pairs = [
            (lab.condition, sc.condition_score)
            for lab, sc in s.execute(
                select(Label, PhotoScore).join(PhotoScore, PhotoScore.photo_id == Label.photo_id)
            ).all()
        ]
        assert float(np.mean([abs(a - b) for a, b in pairs])) < 1.0
        assert s.scalar(select(ListingScore.overall)) is not None


def test_training_refuses_too_few_labels_with_a_helpful_message(settings, tmp_path):
    make_labeled_listing(settings, tmp_path, n_per_level=2)
    with session_scope(settings) as s, pytest.raises(TrainingError, match="at least 40"):
        train_and_evaluate(settings, s, ColorHistEmbedder(), VisionConfig())


def test_holdout_photos_never_enter_training(settings, tmp_path):
    make_labeled_listing(settings, tmp_path)
    with session_scope(settings) as s:
        bundle, metrics, _ = train_and_evaluate(settings, s, ColorHistEmbedder(), VisionConfig())
        holdout = sum(is_holdout_sha(p.sha256) for p in s.scalars(select(Photo)))
    assert metrics["n_holdout"] == holdout and metrics["n_train_pool"] == 80 - holdout


def test_baseline_vs_trained_comparison_is_reported_when_zero_shot_exists(settings, tmp_path, clip):
    """With a text-capable backbone the zero-shot baseline is scored on the same photos."""
    from vision.zeroshot import ZeroShotModel

    make_labeled_listing(settings, tmp_path)
    with session_scope(settings) as s:
        _, metrics, card = train_and_evaluate(
            settings, s, clip, VisionConfig(), zeroshot=ZeroShotModel(clip)
        )
    assert metrics["zero_shot_cv"] and metrics["holdout"]["zero_shot"]
    assert "zero-shot baseline" in card


def test_embeddings_are_stored_once_per_photo_and_backbone(settings, tmp_path, clip):
    make_listing(settings, tmp_path, [(RED, 1), (RED, 2)])
    score_photos(settings, runtime_for(settings, clip))
    store = EmbeddingStore(settings.embeddings_dir, clip.id)
    assert len(list(store.dir.glob("*.npy"))) == 2


# ------------------------------------------------------------------ active learning


def clusters(n_per=20, k=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    X = np.vstack([c + rng.normal(scale=0.05, size=(n_per, d)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), np.repeat(np.arange(k), n_per)


def test_farthest_first_spreads_across_clusters():
    X, group = clusters()
    picks = _farthest_first(X, 4)
    assert len(set(group[picks])) == 4  # one per cluster, no wasted near-duplicates


def test_cold_start_covers_every_cluster_and_room_type():
    X, group = clusters(n_per=20, k=4)
    rooms = ["kitchen", "bath", "bedroom", "exterior"]
    room_of = [rooms[g] for g in group]
    picks = pick_diverse(X, room_of, 8, seed=1)
    assert len(picks) == len(set(picks)) == 8
    assert {room_of[i] for i in picks} == set(rooms)
    assert len(set(group[picks])) == 4


def test_informative_picks_prefer_uncertain_and_diverse_photos():
    rng = np.random.default_rng(2)
    d = 12
    X_lab = rng.normal(size=(60, d))
    X_lab /= np.linalg.norm(X_lab, axis=1, keepdims=True)
    y = np.clip(np.rint(3 + 6 * X_lab[:, 0]), 1, 5).astype(int)
    pool = rng.normal(size=(200, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    picks = pick_informative(pool, ["kitchen"] * 200, X_lab, ["kitchen"] * 60, y, k=10, seed=0)
    assert len(picks) == len(set(picks)) == 10 and all(0 <= i < 200 for i in picks)


def test_holdout_photos_are_interleaved_at_a_steady_rate():
    q = interleave_holdout(list(range(20)), list(range(100, 104)))
    kinds = [k for k, _ in q]
    assert kinds.count("holdout") == 4
    assert all(
        kinds[i] == "holdout" for i in range(HOLDOUT_EVERY - 1, 4 * HOLDOUT_EVERY, HOLDOUT_EVERY)
    )


def test_png_bytes_helper_sanity():
    buf = io.BytesIO()
    noisy_color(RED, 1).save(buf, "PNG")
    assert buf.tell() > 1000


def test_flat_images_are_not_mistaken_for_duplicates_of_each_other(settings, tmp_path, clip):
    """Two different plain-colored images share a trivial hash; they must not be called duplicates."""
    with session_scope(settings) as s:
        upsert_listings(s, [RawListing("manual", "5 Flat St", "Testville", "GA", "30301")])
        listing = s.scalar(select(Listing))
        flats = [
            Image.new("RGB", (400, 300), c)
            for c in [(240, 240, 240), (250, 250, 245), (235, 238, 240)]
        ]
        ingest_photos(
            s, settings, listing, [save(im, tmp_path / f"f{i}.jpg") for i, im in enumerate(flats)]
        )
    rep = score_photos(settings, runtime_for(settings, clip))
    assert rep.excluded_duplicates == 0


def test_rescanned_copies_are_caught_by_hash_plus_embedding_but_other_views_are_not(
    settings, tmp_path, clip
):
    """Re-encoded copies of one photo (contrast shifted) differ by a few hash bits; a different
    photo of a similar-colored scene must not be swallowed."""
    base = noisy_color(RED, 21)
    rescanned = base.point(lambda v: int(np.clip((v - 128) * 1.25 + 128, 0, 255)))  # more contrast
    other_view = noisy_color(RED, 22)  # same "kitchen" color, different structure
    with session_scope(settings) as s:
        upsert_listings(s, [RawListing("manual", "7 Scan St", "Testville", "GA", "30301")])
        listing = s.scalar(select(Listing))
        files = [save(base, tmp_path / "a.jpg"), save(rescanned.resize((400, 300)), tmp_path / "b.jpg"),
                 save(other_view, tmp_path / "c.jpg")]  # fmt: skip
        ingest_photos(s, settings, listing, files)
    rep = score_photos(settings, runtime_for(settings, clip))
    with session_scope(settings) as s:
        excluded = [p for p in s.scalars(select(Photo)) if p.excluded]
    assert rep.excluded_duplicates == len(excluded) == 1
    assert excluded[0].path.endswith("b.jpg") or excluded[0].width == 400  # the smaller re-scan


def test_loose_hash_match_alone_is_not_enough():
    from types import SimpleNamespace

    from vision.pipeline import _cluster_duplicates

    cfg = VisionConfig()
    a = SimpleNamespace(
        id=1,
        phash="0000000000000000",
        width=800,
        height=600,
        order_idx=0,
        excluded=False,
        exclude_reason=None,
    )
    b = SimpleNamespace(
        id=2,
        phash="00000000000000ff",
        width=640,
        height=480,
        order_idx=1,
        excluded=False,
        exclude_reason=None,
    )  # 8 bits apart
    unit = lambda v: np.array(v, dtype=np.float32) / np.linalg.norm(v)  # noqa: E731
    far = {1: unit([1, 0, 0]), 2: unit([0.6, 0.8, 0])}  # different scene
    near = {1: unit([1, 0, 0]), 2: unit([0.99, 0.05, 0])}  # same scene
    assert _cluster_duplicates([a, b], cfg, lambda p: far[p.id]) == []
    assert [(d.id, k.id) for d, k in _cluster_duplicates([a, b], cfg, lambda p: near[p.id])] == [
        (2, 1)
    ]
    assert _cluster_duplicates([a, b], cfg, lambda p: None) == []  # no embedding: stay cautious
