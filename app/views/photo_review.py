"""Photo Review: score a listing's photos, see why, and fix anything the model got wrong."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import select

from app import vision_services as vs
from app.ui import context, disclaimer, thumbnail
from app.vision_ui import get_runtime, not_installed_notice
from core.db import session_scope
from core.models import Listing
from vision.config import FLAG_LABELS, REASON_AUTO_PREFIX, REASON_KEEP, REASON_MANUAL, ROOM_LABELS
from vision.embedders import VisionUnavailable

COLS = 4
FLAG_SHOW_AT = 0.6  # show a flag as "possible" at or above this probability


def render() -> None:
    settings = context()
    st.title("Photo Review")
    disclaimer()
    st.caption(
        "Photos are scored on the physical condition of what's visible (surfaces, fixtures, "
        "damage), 1 = poor to 5 = like new. Floor plans, maps, logos and near-duplicate photos "
        "are set aside automatically. You can override anything."
    )
    with session_scope(settings) as s:
        listings = [
            (lst.id, lst.address_full, len(lst.photos))
            for lst in s.scalars(select(Listing).order_by(Listing.id.desc()))
        ]
    with_photos = [x for x in listings if x[2]]
    if not with_photos:
        st.info("No photos yet. Add some on the Listings page (Photos tab).")
        return
    labels = {i: f"{addr} ({n} photos)" for i, addr, n in with_photos}
    listing_id = st.selectbox("Listing", list(labels), format_func=labels.get)

    ready = not not_installed_notice()
    c1, c2, c3 = st.columns([2, 2, 3])
    rescore = c3.checkbox("Re-score even if already done")
    run_ids: list[int] | None = None
    if c1.button("Score this listing", type="primary", disabled=not ready):
        run_ids = [listing_id]
    if c2.button("Score all listings", disabled=not ready):
        run_ids = [i for i, _, _ in with_photos]
    if run_ids is not None:
        _run(settings, run_ids, rescore)

    report = st.session_state.get("score_report")
    if report:
        st.success(
            f"Scored {report.scored} photo(s) with `{report.model_version}`. Set aside: "
            f"{report.excluded_junk} not of the property, {report.excluded_duplicates} near-duplicates."
            + (f" {report.missing_files} file(s) couldn't be read." if report.missing_files else "")
        )

    review = vs.review_listing(settings, listing_id)
    if review is None:
        return
    _summary(review)
    _grid(settings, review)


def _run(settings, listing_ids: list[int], rescore: bool) -> None:  # noqa: ANN001
    try:
        with st.status("Scoring photos…", expanded=True) as status:
            runtime = get_runtime(settings)
            bar = st.progress(0.0, text="Starting…")

            def progress(stage: str, done: int, total: int) -> None:
                bar.progress(
                    min(done / total, 1.0) if total else 1.0, text=f"{stage}: {done} of {total}"
                )

            report = vs.score_listings(settings, runtime, listing_ids, rescore, progress)
            status.update(label="Done", state="complete")
        for note in runtime.notes:
            st.warning(note)
        st.session_state["score_report"] = report
    except VisionUnavailable as exc:
        st.error(str(exc))
    except OSError as exc:
        st.error(
            "Couldn't load the photo model. The first run downloads it, so check your internet "
            f"connection and free disk space, then try again. ({type(exc).__name__})"
        )


def _summary(review: vs.ListingReview) -> None:
    if review.version is None:
        st.info("This listing hasn't been scored yet. Press “Score this listing”.")
        return
    baseline = review.version.startswith("zeroshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condition (1 to 5)", "n/a" if review.overall is None else f"{review.overall:.1f}")
    worst = (
        "n/a"
        if not review.worst_room
        else f"{ROOM_LABELS[review.worst_room].split(' ')[0]} {review.worst_room_score:.1f}"
    )
    c2.metric("Weakest room", worst)
    c3.metric(
        "Photos used", int(sum(v["n"] for v in review.by_room.values())) if review.by_room else 0
    )
    c4.metric("Scored by", "Baseline" if baseline else "Your model")
    if baseline:
        st.caption(
            "This is the zero-shot baseline: a general-purpose guess with no training on your eye. "
            "Treat it as rough. Rate some photos on the Labeling page and train a model to improve it."
        )
    if review.limited_coverage:
        missing = ", ".join(r for r in review.missing_core_rooms) or "enough photos"
        st.warning(
            f"Limited photo coverage: no usable photo of {missing}. The score may not reflect the whole home."
        )
    if review.by_room:
        st.dataframe(
            [
                {
                    "Room": ROOM_LABELS[r],
                    "Condition": round(v["score"], 1),
                    "Photos": int(v["n"]),
                    "Confidence": f"{v['confidence']:.0%}",
                }
                for r, v in review.by_room.items()
            ],
            hide_index=True,
            width="stretch",
        )


def _toggle_exclude(settings, photo_id: int, key: str) -> None:  # noqa: ANN001
    vs.set_photo_excluded(settings, photo_id, bool(st.session_state.get(key)))


def _reason_text(reason: str | None) -> str:
    if reason == REASON_MANUAL:
        return "You excluded this photo"
    if reason == REASON_KEEP:
        return "Kept by you (the app had set it aside)"
    if reason and reason.startswith(REASON_AUTO_PREFIX):
        return "Set aside: " + reason[len(REASON_AUTO_PREFIX) :]
    return ""


def _grid(settings, review: vs.ListingReview) -> None:  # noqa: ANN001
    weakest_id = None
    if review.worst_room:
        in_room = [
            p
            for p in review.photos
            if p.room == review.worst_room and p.condition is not None and not p.photo.excluded
        ]
        if in_room:
            weakest_id = min(in_room, key=lambda p: p.condition).photo.id
    cols = st.columns(COLS)
    for i, item in enumerate(review.photos):
        p = item.photo
        with cols[i % COLS]:
            thumb = thumbnail(settings, p)
            if thumb is None:
                st.warning("File missing")
            else:
                st.image(str(thumb), width="stretch")
            if p.excluded:
                st.badge(_reason_text(p.exclude_reason) or "Excluded", color="gray")
            if item.condition is not None:
                st.caption(
                    f"**{ROOM_LABELS.get(item.room or 'other', 'Other').split(' ')[0]}**"
                    f" ({(item.room_conf or 0):.0%}) · condition **{item.condition:.1f}**/5"
                    f" ({(item.condition_conf or 0):.0%} sure)"
                )
            if p.id == weakest_id:
                st.badge(f"Weakest photo in {review.worst_room}", color="red")
            for flag, prob in item.flags.items():
                if prob >= FLAG_SHOW_AT and flag in FLAG_LABELS:
                    st.badge(f"possible: {FLAG_LABELS[flag].lower()}", color="orange")
            if item.quality_note:
                st.caption(f"Photo quality: {item.quality_note}")
            key = f"ex_{p.id}"
            st.checkbox(
                "Exclude from scoring",
                value=p.excluded,
                key=key,
                on_change=_toggle_exclude,
                args=(settings, p.id, key),
            )
            st.caption(f"#{p.order_idx + 1} · {p.width}×{p.height}")
