"""Labeling: rate photos 1-5 with one keypress. Your labels train the model."""

from __future__ import annotations

import time

import streamlit as st

from app import vision_services as vs
from app.ui import context, disclaimer, thumbnail
from app.vision_ui import get_runtime, not_installed_notice
from core.db import session_scope
from core.models import Photo
from vision.config import FLAG_LABELS, FLAGS, ROOM_LABELS, ROOMS, RUBRIC_VERSION
from vision.embedders import VisionUnavailable

SCORE_NAMES = {1: "Poor", 2: "Fair", 3: "Average", 4: "Good", 5: "Like new"}
ROOM_KEYS = {
    "kitchen": "K",
    "bath": "B",
    "living": "L",
    "bedroom": "R",
    "exterior": "E",
    "other": "O",
}
FLAG_KEYS = {
    "water_stain": "W",
    "damaged_flooring": "F",
    "dated_finishes": "T",
    "clutter": "C",
    "poor_photo_quality": "P",
}
GOAL = 300  # a good first target; 300-500 is the plan
QUEUE_SIZE = 20

RUBRIC_MD = f"""
**Rate what you can see: surfaces, fixtures and damage. Ignore furniture, staging and the
photographer's skill.** (rubric {RUBRIC_VERSION})

| Score | Meaning |
|---|---|
| **5 Like new** | New or recently renovated; no visible wear; current finishes |
| **4 Good** | Well maintained and clean; maybe lightly dated; no defects visible |
| **3 Average** | Functional with normal wear; dated but intact; minor cosmetic issues |
| **2 Fair** | Visible deferred maintenance; worn or damaged surfaces; needs real work |
| **1 Poor** | Major problems: water damage, severe damage, gutted or unsafe-looking |

What to look at: **kitchen** cabinets, counters, backsplash, appliances, flooring ·
**bath** tile and grout, vanity, tub/shower, caulk, mold · **living/bedroom** flooring, paint,
ceiling, windows · **exterior** roof, siding/paint, gutters, windows, porch, driveway, lawn.

Tick a flag only if you can see it: water stain, damaged flooring, dated finishes, clutter,
poor photo quality. Pick **Unsure** rather than guessing. If it isn't a photo of the property
(floor plan, map, logo), choose **Exclude photo**.
"""

KEYBOARD_JS = """
<script>
(function () {
  if (window.__showingKeys) return;
  window.__showingKeys = true;
  const MAP = {"1":"1","2":"2","3":"3","4":"4","5":"5",
    "k":"K","b":"B","l":"L","r":"R","e":"E","o":"O",
    "w":"W","f":"F","t":"T","c":"C","p":"P","s":"S","u":"U","n":"N","x":"X"};
  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey || ev.repeat) return;
    const t = ev.target, tag = (t.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable) return;
    const token = MAP[ev.key.toLowerCase()];
    if (!token) return;
    const needle = "[" + token + "]";
    for (const el of document.querySelectorAll("button, label")) {
      if ((el.innerText || el.textContent || "").includes(needle)) {
        ev.preventDefault();
        el.click();
        return;
      }
    }
  });
})();
</script>
"""


def render() -> None:
    settings = context()
    st.title("Labeling")
    disclaimer()
    ready = not not_installed_notice()
    stats = vs.label_stats(settings)

    active = vs.active_model_version(settings)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Photos labeled", stats.labeled)
    c2.metric("In the holdout", stats.holdout_labeled, help="Set aside to test the model fairly.")
    c3.metric("Still to label", stats.available)
    c4.metric("Scoring with", "Your model" if active else "Baseline")
    st.progress(
        min(stats.labeled / GOAL, 1.0),
        text=f"{stats.labeled} of about {GOAL} (a good first target)",
    )
    with st.expander("How to rate (rubric)", expanded=stats.labeled == 0):
        st.markdown(RUBRIC_MD)

    if stats.available == 0 and stats.labeled == 0:
        st.info("No photos to label yet. Add photos on the Listings page first.")
        return

    if ready and stats.embedded < stats.available:
        st.info(
            f"{stats.available - stats.embedded} photo(s) haven't been read by the model yet. Reading "
            "them once lets the app show you the most useful photos first."
        )
        if st.button("Prepare photos", type="primary"):
            _prepare(settings)
            st.rerun()
    elif not ready:
        st.caption(
            "Without the photo model, photos are shown in random order (still fine for labeling)."
        )

    _labeling_card(settings, ready)
    st.html(KEYBOARD_JS, unsafe_allow_javascript=True)
    st.divider()
    _train_section(settings, stats, ready)


# ------------------------------------------------------------------ preparing photos


def _prepare(settings) -> None:  # noqa: ANN001
    try:
        with st.status("Reading photos…", expanded=True) as status:
            runtime = get_runtime(settings)
            bar = st.progress(0.0, text="Starting…")
            n = vs.prepare_embeddings(
                settings,
                runtime,
                lambda d, t: bar.progress(min(d / t, 1.0) if t else 1.0, text=f"{d} of {t}"),
            )
            status.update(label=f"Read {n} photos", state="complete")
        st.session_state.pop("lab_queue", None)
    except VisionUnavailable as exc:
        st.error(str(exc))
    except OSError as exc:
        st.error(
            f"Couldn't load the photo model; check your internet connection. ({type(exc).__name__})"
        )


# ------------------------------------------------------------------ the labeling card


def _ensure_queue(settings) -> list[vs.QueueItem]:  # noqa: ANN001
    queue: list[vs.QueueItem] = st.session_state.get("lab_queue", [])
    pos = st.session_state.get("lab_pos", 0)
    if not queue or pos >= len(queue):
        queue = vs.labeling_queue(settings, QUEUE_SIZE)
        st.session_state["lab_queue"], st.session_state["lab_pos"] = queue, 0
    return queue


def _submit(settings, photo_id: int, score: int | None, unsure: bool = False) -> None:  # noqa: ANN001
    room = st.session_state.get(f"room_{photo_id}", "other")
    flags = [f for f in FLAGS if st.session_state.get(f"flag_{f}_{photo_id}")]
    spent = int((time.time() - st.session_state.get("lab_t0", time.time())) * 1000)
    vs.save_label(
        settings, photo_id, condition=score, room=room, flags=flags, unsure=unsure, ms_spent=spent
    )
    st.session_state["lab_last"] = (
        f"{'Unsure' if unsure else score} ({ROOM_LABELS[room].split(' ')[0].lower()})"
    )
    st.session_state["lab_last_room"] = room
    st.session_state["lab_pos"] = st.session_state.get("lab_pos", 0) + 1


def _skip() -> None:
    st.session_state["lab_pos"] = st.session_state.get("lab_pos", 0) + 1


def _exclude(settings, photo_id: int) -> None:  # noqa: ANN001
    vs.exclude_photo(settings, photo_id)
    _skip()


def _undo(settings) -> None:  # noqa: ANN001
    pid = vs.undo_last_label(settings)
    if pid is not None:
        queue = st.session_state.get("lab_queue", [])
        pos = st.session_state.get("lab_pos", 0)
        queue.insert(pos, vs.QueueItem(pid, "Re-label"))
        st.session_state["lab_queue"] = queue
        st.session_state["lab_last"] = "(undone)"


def _labeling_card(settings, ready: bool) -> None:  # noqa: ANN001
    queue = _ensure_queue(settings)
    if not queue:
        st.success(
            "Every available photo has a label. Nice! Add more photos, or train a model below."
        )
        return
    item = queue[st.session_state["lab_pos"]]
    with session_scope(settings) as s:
        photo = s.get(Photo, item.photo_id)
    if photo is None:
        _skip()
        st.rerun()
        return
    if st.session_state.get("lab_t0_pid") != photo.id:
        st.session_state["lab_t0_pid"], st.session_state["lab_t0"] = photo.id, time.time()

    pid = photo.id
    left, right = st.columns([3, 2])
    with left:
        thumb = thumbnail(settings, photo, 1000)
        if thumb is None:
            st.warning("This photo's file is missing.")
        else:
            st.image(str(thumb), width="stretch")
        st.caption(
            f"{item.strategy} · {st.session_state['lab_pos'] + 1} of {len(queue)} in this batch"
            + (
                f" · last: {st.session_state['lab_last']}"
                if st.session_state.get("lab_last")
                else ""
            )
        )
    with right:
        guess_room = st.session_state.get("lab_last_room", "kitchen")
        st.radio(
            "What room is this?",
            ROOMS,
            index=ROOMS.index(guess_room) if guess_room in ROOMS else 0,
            format_func=lambda r: f"{ROOM_LABELS[r]} [{ROOM_KEYS[r]}]",
            key=f"room_{pid}",
        )
        st.markdown("**Condition** (press 1 to 5)")
        cols = st.columns(5)
        for n in range(1, 6):
            cols[n - 1].button(
                f"{n} [{n}]",
                key=f"score_{n}_{pid}",
                on_click=_submit,
                args=(settings, pid, n),
                width="stretch",
                help=SCORE_NAMES[n],
            )
        st.caption(" · ".join(f"{n} {name}" for n, name in SCORE_NAMES.items()))
        st.markdown("**Flags** (only if visible)")
        for f in FLAGS:
            st.checkbox(f"{FLAG_LABELS[f]} [{FLAG_KEYS[f]}]", key=f"flag_{f}_{pid}")
        b1, b2, b3, b4 = st.columns(4)
        b1.button("Unsure [N]", on_click=_submit, args=(settings, pid, None, True), width="stretch")
        b2.button("Skip [S]", on_click=_skip, width="stretch")
        b3.button("Undo [U]", on_click=_undo, args=(settings,), width="stretch")
        b4.button("Exclude photo [X]", on_click=_exclude, args=(settings, pid), width="stretch",
                  help="Not a photo of the property (floor plan, map, logo...)")  # fmt: skip
        if st.toggle(
            "Show the model's guess",
            key=f"guess_{pid}",
            help="Off by default so it can't sway your rating.",
        ):
            _show_guess(settings, ready, pid)


def _show_guess(settings, ready: bool, pid: int) -> None:  # noqa: ANN001
    if not ready:
        st.caption("Needs the photo model.")
        return
    try:
        guess = vs.zero_shot_guess(get_runtime(settings), settings, pid)
    except (VisionUnavailable, OSError):
        guess = None
    if guess is None:
        st.caption("Prepare the photos first to see a guess.")
        return
    st.caption(
        f"Baseline guess: {ROOM_LABELS[guess['room']].split(' ')[0]}, condition about {guess['condition']:.1f}"
    )


# ------------------------------------------------------------------ training


def _train_section(settings, stats: vs.LabelStats, ready: bool) -> None:  # noqa: ANN001
    st.subheader("Train a model from your labels")
    st.caption(
        "Compares two small models on your labels, tests the winner on a holdout set you never "
        "trained on, and reports how it does against the untrained baseline."
    )
    versions = vs.model_versions(settings)
    active = vs.active_model_version(settings)
    if st.button("Train from my labels", disabled=not ready or stats.labeled < 10):
        try:
            with st.status("Training…", expanded=True) as status:
                runtime = get_runtime(settings)
                bar = st.progress(0.0, text="Reading labeled photos…")
                version, metrics = vs.train_model(
                    settings, runtime, lambda d, t: bar.progress(min(d / t, 1.0) if t else 1.0)
                )
                status.update(label="Trained", state="complete")
            st.session_state["trained_version"] = version
            st.rerun()
        except vs.TrainingError as exc:
            st.warning(str(exc))
        except (VisionUnavailable, OSError) as exc:
            st.error(f"Couldn't train: {exc}")
    if stats.labeled < 10:
        st.caption("Label at least a few dozen photos first.")

    for v in versions[:5]:
        _model_row(settings, v, v["version"] == active)


def _fmt(x) -> str:  # noqa: ANN001
    return "n/a" if x is None else f"{x:.2f}"


def _model_row(settings, v: dict, is_active: bool) -> None:  # noqa: ANN001
    name = v["version"]
    best = v["candidates"][v["chosen_head"]]["cv"]
    with st.container(border=True):
        st.markdown(f"**{name}** {'· in use' if is_active else ''}")
        rows = [
            {
                "Evaluated on": "Cross-validation",
                "Photos": best["n"],
                "Avg miss (levels)": _fmt(best["mae"]),
                "Agreement (kappa)": _fmt(best["qwk"]),
            }
        ]
        if v.get("zero_shot_cv"):
            z = v["zero_shot_cv"]
            rows.append(
                {
                    "Evaluated on": "Baseline, same photos",
                    "Photos": z["n"],
                    "Avg miss (levels)": _fmt(z["mae"]),
                    "Agreement (kappa)": _fmt(z["qwk"]),
                }
            )
        if v.get("holdout"):
            h = v["holdout"]
            rows.append(
                {
                    "Evaluated on": "Holdout (never trained on)",
                    "Photos": h["n"],
                    "Avg miss (levels)": _fmt(h["mae"]),
                    "Agreement (kappa)": _fmt(h["qwk"]),
                }
            )
            if h.get("zero_shot"):
                z = h["zero_shot"]
                rows.append(
                    {
                        "Evaluated on": "Baseline on holdout",
                        "Photos": z["n"],
                        "Avg miss (levels)": _fmt(z["mae"]),
                        "Agreement (kappa)": _fmt(z["qwk"]),
                    }
                )
        else:
            st.caption(
                "Not enough holdout labels yet to test it fairly; the first row is cross-validation only."
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        if v.get("test_retest"):
            t = v["test_retest"]
            st.caption(
                f"Your own consistency on {t['pairs']} re-labeled photos: kappa {_fmt(t['qwk'])}."
            )
        a, b = st.columns(2)
        if not is_active and a.button("Use this model for scoring", key=f"act_{name}"):
            vs.activate_model(settings, name)
            st.toast("Now in use. Re-score photos on the Photo Review page.")
            st.rerun()
        if is_active and a.button("Go back to the baseline", key=f"deact_{name}"):
            vs.activate_model(settings, None)
            st.rerun()
        with b.popover("Model card"):
            st.markdown(vs.model_card(settings, name))
