"""Train a condition model from the user's labels, evaluate it honestly, and write a model card.

Honesty rules baked in:
  * 1 photo in 5 (chosen by file hash, fixed forever) is a holdout. It is never used to fit,
    choose, or tune anything. Reported holdout numbers are the ones to believe.
  * Model choice (ridge vs. CORN) uses cross-validation on the training pool only.
  * The zero-shot baseline is scored on the same rows, so you can see whether training helped.
  * Re-labeled photos give your own test-retest agreement, the realistic ceiling for any model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import Settings
from core.models import Label, Photo
from sources.photos import resolve_photo_path
from vision import metrics as M
from vision.config import (
    FLAG_LABELS,
    FLAGS,
    ROOM_LABELS,
    ROOMS,
    RUBRIC_VERSION,
    VisionConfig,
    is_holdout_sha,
    slug,
)
from vision.embedders import Embedder, EmbeddingStore, embed_missing
from vision.heads import (
    CONDITION_HEADS,
    FlagClassifier,
    RoomClassifier,
    make_features,
)
from vision.predict import ModelBundle
from vision.zeroshot import ZeroShotModel

ProgressFn = Callable[[int, int], None]
MIN_HOLDOUT_FOR_REPORT = 10
MIN_ROOM_LABELS = 20
MIN_FLAG_EXAMPLES = 8


class TrainingError(Exception):
    """Not enough (or not usable) labels; the message says what to do."""


def _f(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class LabeledSet:
    photo_ids: list[int]
    shas: list[str]
    X: np.ndarray  # (n, d) embeddings
    rooms: list[str]  # as labeled by the user
    y: np.ndarray  # condition 1..5, or 0 when the user marked "unsure"
    flags: np.ndarray  # (n, len(FLAGS)) 0/1
    holdout: np.ndarray  # bool
    retest_pairs: list[tuple[int, int]]  # (first label, latest label) for photos labeled twice

    @property
    def scored(self) -> np.ndarray:
        return self.y > 0


def collect_labeled(
    settings: Settings,
    session: Session,
    store: EmbeddingStore,
    embedder: Embedder,
    cfg: VisionConfig,
    progress: ProgressFn | None = None,
) -> LabeledSet:
    """Latest label per photo, joined with embeddings (computed and cached if missing)."""
    labels = list(session.scalars(select(Label).order_by(Label.photo_id, Label.id)))
    by_photo: dict[int, list[Label]] = {}
    for lab in labels:
        by_photo.setdefault(lab.photo_id, []).append(lab)
    photos = {
        p.id: p for p in session.scalars(select(Photo).where(Photo.id.in_(list(by_photo) or [0])))
    }
    items = [
        (photos[pid].sha256, resolve_photo_path(settings, photos[pid].path))
        for pid in by_photo
        if pid in photos
    ]
    vecs, _failed = embed_missing(store, embedder, items, cfg, progress)

    ids, shas, X, rooms, y, flags, hold, pairs = [], [], [], [], [], [], [], []
    for pid, labs in by_photo.items():
        p = photos.get(pid)
        if p is None or p.sha256 not in vecs:
            continue
        latest = labs[-1]
        ids.append(pid)
        shas.append(p.sha256)
        X.append(vecs[p.sha256])
        rooms.append(latest.room_type or "other")
        y.append(0 if latest.unsure or latest.condition is None else int(latest.condition))
        row = latest.flags or []
        flags.append([1 if f in row else 0 for f in FLAGS])
        hold.append(is_holdout_sha(p.sha256, cfg.holdout_fraction_denominator))
        scored = [lab for lab in labs if not lab.unsure and lab.condition is not None]
        if len(scored) >= 2:
            pairs.append((int(scored[0].condition), int(scored[-1].condition)))
    dim = X[0].shape[0] if X else 0
    return LabeledSet(
        ids, shas,
        np.stack(X) if X else np.zeros((0, dim), np.float32),
        rooms, np.array(y, dtype=int),
        np.array(flags, dtype=int).reshape(len(ids), len(FLAGS)),
        np.array(hold, dtype=bool), pairs,
    )  # fmt: skip


# ------------------------------------------------------------------ cross-validation


def _oof(head_name: str, Xf: np.ndarray, y: np.ndarray, splits: int, seed: int):  # noqa: ANN202
    from sklearn.model_selection import KFold

    score, klass = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    for tr, va in KFold(splits, shuffle=True, random_state=seed).split(Xf):
        head = CONDITION_HEADS[head_name]().fit(Xf[tr], y[tr], seed=seed)
        score[va], klass[va] = head.predict_score(Xf[va]), head.predict_class(Xf[va])
    return score, klass


def _summary(y, score, klass) -> dict:
    return {
        "n": int(len(y)),
        "mae": _f(M.mae(y, score)),
        "qwk": _f(M.quadratic_weighted_kappa(y, klass)),
    }


def _round_class(score: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(score), 1, 5).astype(int)


def train_and_evaluate(
    settings: Settings,
    session: Session,
    embedder: Embedder,
    cfg: VisionConfig,
    zeroshot: ZeroShotModel | None = None,
    progress: ProgressFn | None = None,
    seed: int = 0,
) -> tuple[ModelBundle, dict, str]:
    store = EmbeddingStore(settings.embeddings_dir, embedder.id)
    ls = collect_labeled(settings, session, store, embedder, cfg, progress)
    pool = ls.scored & ~ls.holdout
    hold = ls.scored & ls.holdout
    n_pool, n_hold = int(pool.sum()), int(hold.sum())
    if n_pool < cfg.min_labels_to_train:
        raise TrainingError(
            f"Label more photos before training: {n_pool} usable labels so far, and training "
            f"needs at least {cfg.min_labels_to_train} (about 1 in 5 of your labels is set aside "
            f"as a permanent holdout, so label roughly {int(cfg.min_labels_to_train * 1.25)} in "
            "total). Aim for 300-500 for reliable results."
        )

    Xf = make_features(ls.X, ls.rooms)
    ip = np.flatnonzero(pool)
    splits = max(3, min(5, n_pool // 8))

    # ---- choose the condition head by cross-validation on the training pool only
    candidates, oofs = {}, {}
    for name in CONDITION_HEADS:
        score, klass = _oof(name, Xf[ip], ls.y[ip], splits, seed)
        oofs[name] = (score, klass)
        candidates[name] = {"cv": _summary(ls.y[ip], score, klass)}

    def rank_key(name: str) -> tuple[float, float]:
        s = candidates[name]["cv"]
        return (s["qwk"] if s["qwk"] is not None else -1.0, -(s["mae"] or 9.0))

    chosen = max(candidates, key=rank_key)

    notes: list[str] = []
    zs_cv = None
    zs_rows = None
    if zeroshot is not None:
        zs_rows = zeroshot.predict(ls.X)
        zscore = np.array([o.condition for o in zs_rows])
        zs_cv = _summary(ls.y[ip], zscore[ip], _round_class(zscore[ip]))
    else:
        notes.append("No zero-shot baseline: this backbone cannot read text prompts.")

    # ---- final fit on the training pool, then the one honest look at the holdout
    head = CONDITION_HEADS[chosen]().fit(Xf[ip], ls.y[ip], seed=seed)
    holdout_report = None
    if n_hold >= MIN_HOLDOUT_FOR_REPORT:
        ih = np.flatnonzero(hold)
        hs, hc = head.predict_score(Xf[ih]), head.predict_class(Xf[ih])
        yh = ls.y[ih]
        holdout_report = {
            **_summary(yh, hs, hc),
            "confusion": M.confusion(yh, hc, [1, 2, 3, 4, 5]).tolist(),
            "per_room_mae": {
                r: [_f(v[0]), v[1]]
                for r, v in M.per_group_mae(list(yh), list(hs), [ls.rooms[i] for i in ih]).items()
            },
            "zero_shot": (
                _summary(
                    yh,
                    np.array([zs_rows[i].condition for i in ih]),
                    _round_class(np.array([zs_rows[i].condition for i in ih])),
                )
                if zs_rows
                else None
            ),  # fmt: skip
        }
    else:
        notes.append(
            f"Only {n_hold} holdout labels so far (need {MIN_HOLDOUT_FOR_REPORT}); results below "
            "are cross-validation only. Keep labeling, and use 'Random check' photos to fill the holdout."
        )

    # ---- test-retest: how consistent is the labeler with themselves?
    retest = None
    if len(ls.retest_pairs) >= MIN_HOLDOUT_FOR_REPORT:
        a, b = zip(*ls.retest_pairs, strict=True)
        retest = {"pairs": len(a), "qwk": _f(M.quadratic_weighted_kappa(a, b)),
                  "mae": _f(M.mae(a, b))}  # fmt: skip
    else:
        notes.append(
            "Re-label about 10% of photos a week later to measure your own consistency; that is "
            "the realistic ceiling for any model."
        )

    # ---- room type
    room_clf, room_report = None, {"trained": False}
    rooms_pool = [ls.rooms[i] for i in ip]
    if len(ip) >= MIN_ROOM_LABELS and len(set(rooms_pool)) >= 2:
        from sklearn.model_selection import KFold

        pred_rooms = np.empty(len(ip), dtype=object)
        for tr, va in KFold(splits, shuffle=True, random_state=seed).split(ls.X[ip]):
            if len({rooms_pool[i] for i in tr}) < 2:
                pred_rooms[va] = "other"
                continue
            c = RoomClassifier().fit(ls.X[ip][tr], [rooms_pool[i] for i in tr], seed=seed)
            pred_rooms[va] = [ROOMS[j] for j in c.predict_proba(ls.X[ip][va]).argmax(axis=1)]
        room_clf = RoomClassifier().fit(ls.X[ip], rooms_pool, seed=seed)
        room_report = {
            "trained": True,
            "cv_accuracy": _f(float(np.mean(np.array(rooms_pool, dtype=object) == pred_rooms))),
            "confusion": M.confusion(rooms_pool, list(pred_rooms), ROOMS).tolist(),
            "zero_shot_accuracy": (
                _f(float(np.mean([zs_rows[i].room == ls.rooms[i] for i in ip])))
                if zs_rows
                else None
            ),
        }
        if n_hold >= MIN_HOLDOUT_FOR_REPORT:
            ih = np.flatnonzero(hold)
            ph = [ROOMS[j] for j in room_clf.predict_proba(ls.X[ih]).argmax(axis=1)]
            room_report["holdout_accuracy"] = _f(
                float(np.mean([ph[k] == ls.rooms[i] for k, i in enumerate(ih)]))
            )
    else:
        notes.append("Room-type classifier not trained (too few labeled photos); using zero-shot.")

    # ---- flags
    flag_clfs, flag_report = {}, {}
    for j, flag in enumerate(FLAGS):
        pos = int(ls.flags[ip, j].sum())
        neg = len(ip) - pos
        flag_report[flag] = {"positives": pos, "trained": False}
        if pos >= MIN_FLAG_EXAMPLES and neg >= MIN_FLAG_EXAMPLES:
            flag_clfs[flag] = FlagClassifier().fit(ls.X[ip], ls.flags[ip, j], seed=seed)
            flag_report[flag]["trained"] = True

    # ---- learning curve (is more labeling still helping?)
    rng = np.random.default_rng(seed)
    curve = []
    for frac in (0.4, 0.6, 0.8, 1.0):
        m = max(20, int(len(ip) * frac))
        if m > len(ip):
            continue
        sub = rng.choice(len(ip), size=m, replace=False)
        s, k = _oof(chosen, Xf[ip][sub], ls.y[ip][sub], 3, seed)
        curve.append({"n": int(m), **_summary(ls.y[ip][sub], s, k)})

    version = f"head-{slug(embedder.id)[:40]}-{datetime.now():%Y%m%d-%H%M%S}"
    created = datetime.now().isoformat(timespec="seconds")
    metrics = {
        "created": created,
        "backbone": embedder.id,
        "rubric_version": RUBRIC_VERSION,
        "chosen_head": chosen,
        "n_labeled": int(len(ls.y)),
        "n_unsure": int((ls.y == 0).sum()),
        "n_train_pool": n_pool,
        "n_holdout": n_hold,
        "label_counts": {str(k): int((ls.y[ip] == k).sum()) for k in range(1, 6)},
        "room_counts": {r: int(sum(1 for x in rooms_pool if x == r)) for r in ROOMS},
        "candidates": candidates,
        "zero_shot_cv": zs_cv,
        "holdout": holdout_report,
        "test_retest": retest,
        "room_type": room_report,
        "flags": flag_report,
        "learning_curve": curve,
        "notes": notes,
    }
    bundle = ModelBundle(
        version=version, backbone_id=embedder.id, condition_head=head, room_clf=room_clf,
        flag_clfs=flag_clfs, created=created,
        meta={"chosen_head": chosen, "n_train_pool": n_pool, "rubric_version": RUBRIC_VERSION},
    )  # fmt: skip
    return bundle, metrics, render_model_card(version, metrics)


# ------------------------------------------------------------------ model card


def _fmt(x, nd=2) -> str:  # noqa: ANN001
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_model_card(version: str, m: dict) -> str:
    lines = [
        f"# Model card: {version}",
        "",
        f"Created {m['created']} with the **{m['backbone']}** backbone and a "
        f"**{m['chosen_head']}** condition head (rubric {m['rubric_version']}).",
        "",
        "## Intended use",
        "An aid for a real estate agent to prioritize which listings to look at. It scores the "
        "*visible physical condition of surfaces in photos* on a 1-5 scale. It is **not an "
        "appraisal or inspection**, and it must not be used to describe or rank neighborhoods or "
        "the people who live in them. The model sees only pixels: never addresses, prices, "
        "listing text or any location or demographic information.",
        "",
        "## Data",
        f"- Labeled photos: {m['n_labeled']} ({m['n_train_pool']} used for training, {m['n_holdout']} "
        f"in the permanent holdout, {m['n_unsure']} marked unsure)",
        "- Condition label counts (training pool): "
        + ", ".join(f"{k}: {v}" for k, v in m["label_counts"].items()),
        "- Room types (training pool): "
        + ", ".join(f"{ROOM_LABELS[r].split(' ')[0]}: {v}" for r, v in m["room_counts"].items()),
        "- Labeled by a single expert (the agent). One labeler means the model learns that "
        "person's judgment, including their quirks.",
        "",
        "## Results",
        "| Model | Photos | MAE (levels) | Quadratic weighted kappa |",
        "|---|---|---|---|",
    ]
    for name, c in m["candidates"].items():
        mark = " (chosen)" if name == m["chosen_head"] else ""
        lines.append(
            f"| {name} cross-validation{mark} | {c['cv']['n']} | {_fmt(c['cv']['mae'])} | {_fmt(c['cv']['qwk'])} |"
        )
    if m["zero_shot_cv"]:
        z = m["zero_shot_cv"]
        lines.append(
            f"| zero-shot baseline, same photos | {z['n']} | {_fmt(z['mae'])} | {_fmt(z['qwk'])} |"
        )
    h = m["holdout"]
    if h:
        lines.append(
            f"| **holdout** (chosen head) | {h['n']} | {_fmt(h['mae'])} | {_fmt(h['qwk'])} |"
        )
        if h.get("zero_shot"):
            z = h["zero_shot"]
            lines.append(
                f"| **holdout** zero-shot baseline | {z['n']} | {_fmt(z['mae'])} | {_fmt(z['qwk'])} |"
            )
    else:
        lines.append("| holdout | not enough holdout labels yet | | |")
    lines += [
        "",
        "MAE is the average miss in condition levels (lower is better). Kappa is 1 for "
        "perfect agreement and 0 for chance (higher is better).",
        "",
    ]
    if m["test_retest"]:
        t = m["test_retest"]
        lines += [
            f"**Your own consistency** on {t['pairs']} re-labeled photos: kappa {_fmt(t['qwk'])}, "
            f"MAE {_fmt(t['mae'])}. A model cannot be expected to agree with you more than you agree with yourself.",
            "",
        ]
    if h:
        lines += ["### Holdout confusion (rows = your label, columns = model, levels 1-5)", "```"]
        lines += [" ".join(f"{v:3d}" for v in row) for row in h["confusion"]]
        lines += [
            "```",
            "",
            "### Holdout error by room type",
            "| Room | MAE | Photos |",
            "|---|---|---|",
        ]
        lines += [f"| {r} | {_fmt(v[0])} | {v[1]} |" for r, v in h["per_room_mae"].items()]
        lines.append("")
    rt = m["room_type"]
    if rt.get("trained"):
        lines += [
            "### Room-type classifier",
            f"- Cross-validated accuracy: {_fmt(rt['cv_accuracy'])}"
            + (
                f" (zero-shot: {_fmt(rt['zero_shot_accuracy'])})"
                if rt.get("zero_shot_accuracy") is not None
                else ""
            ),
        ]
        if rt.get("holdout_accuracy") is not None:
            lines.append(f"- Holdout accuracy: {_fmt(rt['holdout_accuracy'])}")
        lines.append("")
    trained = [FLAG_LABELS[f] for f, v in m["flags"].items() if v["trained"]]
    lines += [
        "### Flags",
        "- Trained from your labels: "
        + (", ".join(trained) or "none yet")
        + ". Others use zero-shot guesses and should be read as 'possible', not fact.",
        "",
    ]
    if m["learning_curve"]:
        lines += [
            "### Learning curve (does more labeling still help?)",
            "| Labels | MAE | Kappa |",
            "|---|---|---|",
        ]
        lines += [
            f"| {c['n']} | {_fmt(c['mae'])} | {_fmt(c['qwk'])} |" for c in m["learning_curve"]
        ]
        lines += [
            "",
            "If the error is still falling steeply, label more. If it has flattened, "
            "the limit is label noise or the frozen features, not quantity. Consider partial "
            "fine-tuning only once you have roughly 1,500 labels *and* the curve is still dropping.",
            "",
        ]
    lines += [
        "## Limitations",
        "- Photos can be staged, virtually staged, edited or flattering; the model judges what it sees.",
        "- A photo shows one angle; a listing score is only as good as the photos provided. Listings "
        "missing kitchen, bath or exterior photos are flagged as limited coverage.",
        "- Lighting, camera and photographer style can shift scores. Treat differences of a few "
        "tenths of a level as noise.",
        "- Confidence values are rough: they reflect model agreement, not a guarantee.",
        "",
    ]
    if m["notes"]:
        lines += ["## Notes"] + [f"- {n}" for n in m["notes"]] + [""]
    return "\n".join(lines)


def models_dir_for(settings: Settings) -> Path:
    return settings.models_dir
