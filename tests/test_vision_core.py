"""Vision building blocks that need no model download: metrics, imaging, heads, rollup, registry."""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("sklearn")

from vision import metrics  # noqa: E402
from vision.config import (  # noqa: E402
    ROOMS,
    VisionConfig,
    is_holdout_sha,
    load_room_weights,
    save_room_weights,
)
from vision.embedders import ColorHistEmbedder, EmbeddingStore, embed_missing  # noqa: E402
from vision.heads import (  # noqa: E402
    CornLinearHead,
    FlagClassifier,
    RidgeThresholdHead,
    RoomClassifier,
    fit_thresholds,
    make_features,
)
from vision.imaging import (  # noqa: E402
    dhash,
    hamming_hex,
    load_rgb,
    poor_quality_reason,
    quality_metrics,
)
from vision.predict import ModelBundle  # noqa: E402
from vision.registry import (  # noqa: E402
    active_version,
    list_versions,
    load_bundle,
    save_version,
    set_active,
)
from vision.rollup import PhotoRoll, rollup  # noqa: E402

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ metrics


def test_qwk_known_values():
    y = [1, 2, 3, 4, 5, 3, 2]
    assert metrics.quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    off_by_one = [min(5, v + 1) if v < 5 else 4 for v in y]
    far = [6 - v for v in y]  # reversed scale
    assert 0.5 < metrics.quadratic_weighted_kappa(y, off_by_one) < 1.0
    assert metrics.quadratic_weighted_kappa(y, far) < 0
    assert np.isnan(metrics.quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]))  # undefined


def test_mae_and_group_mae_and_confusion():
    assert metrics.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    per = metrics.per_group_mae([1, 3, 5, 5], [1, 4, 5, 3], ["a", "a", "b", "b"])
    assert per["a"] == (0.5, 2) and per["b"] == (1.0, 2)
    cm = metrics.confusion([1, 1, 2], [1, 2, 2], [1, 2, 3])
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]


# ------------------------------------------------------------------ imaging


def solid(color, size=(400, 300)):
    return Image.new("RGB", size, color)


def noisy(size=(400, 300), seed=1):
    a = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(a)


def test_exif_rotation_is_applied(tmp_path):
    im = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif)
    assert load_rgb(p).size == (100, 200)
    assert load_rgb(p, max_side=50).size[1] == 50


def test_dhash_groups_near_duplicates_and_separates_different_photos():
    a = noisy(seed=1)
    resized = a.resize((200, 150)).resize((400, 300))
    brighter = a.point(lambda v: min(255, v + 6))
    other = noisy(seed=2)
    assert hamming_hex(dhash(a), dhash(a)) == 0
    assert hamming_hex(dhash(a), dhash(resized)) <= 8
    assert hamming_hex(dhash(a), dhash(brighter)) <= 4
    assert hamming_hex(dhash(a), dhash(other)) > 16


def test_quality_flags():
    cfg = VisionConfig()
    assert poor_quality_reason(quality_metrics(noisy()), cfg) is None
    assert poor_quality_reason(quality_metrics(solid((128, 128, 128))), cfg) == "blurry"
    assert poor_quality_reason(quality_metrics(noisy((100, 80))), cfg) == "low resolution"
    dark = noisy().point(lambda v: v // 10)
    assert poor_quality_reason(quality_metrics(dark), cfg) in {"too dark", "blurry"}


def test_holdout_assignment_is_stable_and_about_a_fifth():
    shas = [f"{i:08x}" + "0" * 56 for i in range(1000)]
    flags = [is_holdout_sha(s) for s in shas]
    assert flags == [is_holdout_sha(s) for s in shas]
    assert 0.15 < sum(flags) / len(flags) < 0.25


# ------------------------------------------------------------------ embeddings cache


def test_embedding_cache_computes_once_and_survives_bad_files(tmp_path):
    for i, c in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        solid(c).save(tmp_path / f"{i}.png")
    (tmp_path / "bad.png").write_bytes(b"nope")
    items = [(f"sha{i}", tmp_path / f"{i}.png") for i in range(3)] + [
        ("shabad", tmp_path / "bad.png")
    ]
    emb, store = ColorHistEmbedder(), EmbeddingStore(tmp_path / "emb", ColorHistEmbedder.id)
    cfg = VisionConfig(batch_size=2)
    seen = []
    vecs, failed = embed_missing(store, emb, items, cfg, progress=lambda d, t: seen.append((d, t)))
    assert set(vecs) == {"sha0", "sha1", "sha2"} and failed == ["shabad"]
    assert seen[-1] == (4, 4)
    calls = []
    real = emb.embed_images
    emb.embed_images = lambda imgs: calls.append(len(imgs)) or real(imgs)
    vecs2, _ = embed_missing(store, emb, items[:3], cfg)
    assert calls == [] and np.allclose(vecs["sha0"], vecs2["sha0"], atol=1e-3)  # all cache hits
    assert all(abs(np.linalg.norm(v) - 1) < 1e-2 for v in vecs.values())


# ------------------------------------------------------------------ heads


def synthetic(n=240, d=24, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    latent = X @ w * 6 + rng.normal(scale=0.25, size=n)
    y = np.clip(np.digitize(latent, np.quantile(latent, [0.1, 0.3, 0.6, 0.85])) + 1, 1, 5)
    return X.astype(np.float32), y


@pytest.mark.parametrize("Head", [RidgeThresholdHead, CornLinearHead])
def test_ordinal_heads_learn_and_are_ordered(Head):
    X, y = synthetic()
    tr, te = slice(0, 180), slice(180, None)
    head = Head().fit(X[tr], y[tr])
    pred = head.predict_class(X[te])
    assert metrics.quadratic_weighted_kappa(y[te], pred) > 0.6
    assert metrics.mae(y[te], head.predict_score(X[te])) < 0.8
    s = head.predict_score(X[te])
    assert s.min() >= 1 and s.max() <= 5
    c = head.confidence(X[te])
    assert c.min() >= 0 and c.max() <= 1


def test_corn_probabilities_are_valid_and_rank_consistent():
    X, y = synthetic()
    head = CornLinearHead().fit(X, y)
    p = head.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p >= -1e-9).all()
    # rank consistency: P(y > k) never increases with k
    gt = 1 - np.cumsum(p, axis=1)[:, :4]
    assert (np.diff(gt, axis=1) <= 1e-9).all()


def test_corn_handles_missing_levels_with_tiny_data():
    X, _ = synthetic(n=30)
    y = np.array([2, 3] * 15)  # levels 1, 4, 5 never seen
    head = CornLinearHead().fit(X, y)
    p = head.predict_proba(X)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_fit_thresholds_improve_agreement_over_default_cuts():
    rng = np.random.default_rng(3)
    y = rng.integers(1, 6, 300)
    pred = y * 0.6 + 1.2 + rng.normal(scale=0.15, size=300)  # compressed scale
    default = metrics.quadratic_weighted_kappa(y, np.digitize(pred, [1.5, 2.5, 3.5, 4.5]) + 1)
    tuned = metrics.quadratic_weighted_kappa(y, np.digitize(pred, fit_thresholds(pred, y)) + 1)
    assert tuned > default


def test_room_and_flag_classifiers():
    rng = np.random.default_rng(5)
    centers = rng.normal(size=(3, 16))
    rooms = ["kitchen", "bath", "bedroom"]
    labels = [rooms[i % 3] for i in range(90)]
    X = np.stack([centers[i % 3] + rng.normal(scale=0.2, size=16) for i in range(90)]).astype(
        np.float32
    )
    clf = RoomClassifier().fit(X, labels)
    probs = clf.predict_proba(X)
    assert probs.shape == (90, len(ROOMS))
    assert [ROOMS[i] for i in probs.argmax(axis=1)] == labels
    assert probs[:, ROOMS.index("exterior")].sum() == 0  # never seen in training
    two = RoomClassifier().fit(
        X[labels_mask(labels, rooms[:2])], [r for r in labels if r in rooms[:2]]
    )
    assert two.predict_proba(X[:4]).shape == (4, len(ROOMS))  # binary case works too
    flag = FlagClassifier().fit(X, np.array([r == "bath" for r in labels]))
    p = flag.predict_proba(X)
    assert p[[i for i, r in enumerate(labels) if r == "bath"]].mean() > 0.8


def labels_mask(labels, keep):
    return np.array([r in keep for r in labels])


# ------------------------------------------------------------------ registry (no pickle)


def test_bundle_round_trips_without_pickle(tmp_path):
    X, y = synthetic(n=120, d=16)
    rooms = [ROOMS[i % 3] for i in range(120)]
    feats = make_features(X, rooms)
    for head in (RidgeThresholdHead().fit(feats, y), CornLinearHead().fit(feats, y)):
        flag = FlagClassifier().fit(X, y > 3)
        bundle = ModelBundle(
            version=f"t-{head.kind}",
            backbone_id="colorhist",
            condition_head=head,
            room_clf=RoomClassifier().fit(X, rooms),
            flag_clfs={"clutter": flag},
            meta={"n": 120},
        )
        save_version(tmp_path, bundle, {"created": "2026-10-08T10:00:00"}, "# card")
        back = load_bundle(tmp_path, bundle.version)
        assert np.allclose(
            back.condition_head.predict_score(feats), head.predict_score(feats), atol=1e-5
        )
        assert np.allclose(
            back.room_clf.predict_proba(X), bundle.room_clf.predict_proba(X), atol=1e-6
        )
        assert np.allclose(
            back.flag_clfs["clutter"].predict_proba(X), flag.predict_proba(X), atol=1e-6
        )
        assert back.meta == {"n": 120} and back.condition_head.kind == head.kind
    assert not list(tmp_path.rglob("*.pkl")) and not list(tmp_path.rglob("*.joblib"))


def test_registry_active_version_and_listing(tmp_path):
    X, y = synthetic(n=60, d=8)
    b = ModelBundle("v1", "colorhist", RidgeThresholdHead().fit(X, y))
    save_version(tmp_path, b, {"created": "2026-10-08T10:00:00"}, "card")
    assert active_version(tmp_path) is None
    set_active(tmp_path, "v1")
    assert active_version(tmp_path) == "v1"
    assert [m["version"] for m in list_versions(tmp_path)] == ["v1"]
    with pytest.raises(ValueError):
        set_active(tmp_path, "missing")
    with pytest.raises(ValueError):
        set_active(tmp_path, "../escape")
    set_active(tmp_path, None)
    assert active_version(tmp_path) is None


# ------------------------------------------------------------------ rollup


W = {"kitchen": 3.0, "bath": 2.5, "exterior": 2.0, "living": 1.5, "bedroom": 1.0, "other": 0.5}


def test_rollup_weights_rooms_and_names_the_worst():
    photos = [
        PhotoRoll("kitchen", 2.0, 0.9), PhotoRoll("kitchen", 2.2, 0.9),
        PhotoRoll("bath", 4.0, 0.9), PhotoRoll("exterior", 4.5, 0.9),
        PhotoRoll("bedroom", 5.0, 0.9), PhotoRoll("bedroom", 5.0, 0.9), PhotoRoll("bedroom", 5.0, 0.9),
    ]  # fmt: skip
    r = rollup(photos, W)
    assert r.worst_room == "kitchen" and r.worst_room_score == pytest.approx(2.1)
    # weighted mean: (2.1*3 + 4*2.5 + 4.5*2 + 5*1) / (3+2.5+2+1)
    assert r.overall == pytest.approx((2.1 * 3 + 4 * 2.5 + 4.5 * 2 + 5 * 1) / 8.5, abs=1e-3)
    assert r.by_room["bedroom"]["n"] == 3 and not r.missing_core_rooms
    assert r.n_photos_used == 7 and not r.limited_coverage


def test_rollup_five_great_bedrooms_cannot_hide_a_bad_kitchen():
    bedrooms = [PhotoRoll("bedroom", 5.0, 1.0)] * 5
    kitchen = [PhotoRoll("kitchen", 1.5, 1.0)]
    r = rollup(bedrooms + kitchen, W)
    assert r.worst_room == "kitchen" and r.overall < 3.5


def test_rollup_flags_limited_coverage_and_handles_edge_cases():
    r = rollup([PhotoRoll("bedroom", 4.0, 0.8)], W)
    assert r.missing_core_rooms == ["kitchen", "bath", "exterior"] and r.limited_coverage
    empty = rollup([], W)
    assert empty.overall is None and empty.worst_room is None
    zero_weight = rollup([PhotoRoll("kitchen", 3.0, 1.0)], {**W, "kitchen": 0.0})
    assert zero_weight.overall is None  # ignoring the only room leaves nothing to score
    only_other = rollup([PhotoRoll("other", 2.0, 1.0)], W)
    assert only_other.worst_room is None and only_other.overall == pytest.approx(2.0)


def test_rollup_trusts_confident_photos_more():
    r = rollup([PhotoRoll("kitchen", 5.0, 1.0), PhotoRoll("kitchen", 1.0, 0.0)], W)
    assert r.by_room["kitchen"]["score"] > 3.5  # the zero-confidence photo still counts a bit


def test_room_weights_can_be_overridden_per_user(tmp_path):
    base = load_room_weights(None)
    assert base["kitchen"] > base["bedroom"]
    save_room_weights(tmp_path, {**base, "kitchen": 5.0, "bedroom": -3})
    mine = load_room_weights(tmp_path)
    assert mine["kitchen"] == 5.0 and mine["bedroom"] == 0.0  # negative values clamp to 0
