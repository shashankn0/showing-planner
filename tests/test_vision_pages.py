"""The Labeling, Photo Review and Settings pages driven like a user would."""

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select
from streamlit.testing.v1 import AppTest

pytest.importorskip("sklearn")

from app import services  # noqa: E402
from app import vision_services as vs  # noqa: E402
from core.config import Settings  # noqa: E402
from core.db import init_db, session_scope  # noqa: E402
from core.models import Label, Listing, ListingScore, Photo, PhotoScore  # noqa: E402
from core.store import upsert_listings  # noqa: E402
from ranking.preferences import Preferences  # noqa: E402
from sources.base import RawListing  # noqa: E402
from sources.photos import ingest_photos  # noqa: E402
from tests.fakes import FakeClip, bath_prompts, floor_plan_prompts, kitchen_prompts  # noqa: E402
from vision.config import REASON_KEEP, VisionConfig  # noqa: E402
from vision.pipeline import score_photos  # noqa: E402
from vision.runtime import build_runtime  # noqa: E402

LABELING = "from app.views.labeling import render\nrender()"
REVIEW = "from app.views.photo_review import render\nrender()"
SETTINGS = "from app.views.settings import render\nrender()"
RED, BLUE, GRAY = (200, 30, 30), (30, 30, 200), (128, 128, 128)


def photo(color, seed, size=(420, 320)):
    rng = np.random.default_rng(seed)
    blocks = rng.normal(0, 18, (6, 8, 1))
    pattern = np.kron(blocks, np.ones((size[1] // 6 + 1, size[0] // 8 + 1, 1)))[
        : size[1], : size[0]
    ]
    arr = np.clip(np.array(color) + pattern + rng.normal(0, 5, (size[1], size[0], 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    return s


@pytest.fixture
def seeded(settings, tmp_path):
    """One listing with 6 distinct photos (kitchens, baths, one 'floor plan')."""
    with session_scope(settings) as s:
        upsert_listings(
            s,
            [RawListing("manual", "1 Page St", "Testville", "GA", "30301", source_listing_id="P1")],
        )
        listing = s.scalar(select(Listing))
        files = []
        for i, color in enumerate([RED, RED, RED, BLUE, BLUE, GRAY]):
            p = tmp_path / f"p_{color[0]}_{i}.jpg"
            photo(color, 10 + i).save(p, "JPEG", quality=92)
            files.append(p)
        ingest_photos(s, settings, listing, files)
    return settings


def scored(settings):
    clip = FakeClip()
    clip.mark(RED, kitchen_prompts())
    clip.mark(BLUE, bath_prompts())
    clip.mark(GRAY, floor_plan_prompts())
    rt = build_runtime(settings, VisionConfig(), embedder=clip)
    return score_photos(settings, rt)


def button(at, text):
    return next(b for b in at.button if text in b.label)


# ------------------------------------------------------------------ labeling


def test_labeling_page_renders_with_no_photos(settings):
    at = AppTest.from_string(LABELING, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("No photos to label yet" in i.value for i in at.info)


def test_a_score_button_saves_a_label_and_advances(seeded):
    at = AppTest.from_string(LABELING, default_timeout=60).run()
    assert not at.exception, at.exception
    first_pid = int(next(r.key for r in at.radio if r.key.startswith("room_")).split("_")[1])
    at.radio(key=f"room_{first_pid}").set_value("bath")
    at.checkbox(key=f"flag_clutter_{first_pid}").check()
    button(at, "4 [4]").click().run()
    assert not at.exception, at.exception
    with session_scope(seeded) as s:
        lab = s.scalar(select(Label))
        assert (lab.photo_id, lab.condition, lab.room_type, lab.flags) == (
            first_pid,
            4,
            "bath",
            ["clutter"],
        )
        assert lab.unsure is False and lab.ms_spent is not None
    assert vs.label_stats(seeded).labeled == 1
    # the next photo is a different one, and the room choice carried over as the new default
    next_pid = int(next(r.key for r in at.radio if r.key.startswith("room_")).split("_")[1])
    assert next_pid != first_pid
    assert at.radio(key=f"room_{next_pid}").value == "bath"


def test_unsure_skip_undo_and_exclude(seeded):
    at = AppTest.from_string(LABELING, default_timeout=60).run()
    pid = lambda: int(next(r.key for r in at.radio if r.key.startswith("room_")).split("_")[1])  # noqa: E731
    a = pid()
    button(at, "Unsure [N]").click().run()
    with session_scope(seeded) as s:
        lab = s.scalar(select(Label))
        assert lab.unsure and lab.condition is None and lab.photo_id == a
    assert vs.label_stats(seeded).unsure == 1 and vs.label_stats(seeded).labeled == 0

    b = pid()
    button(at, "Skip [S]").click().run()
    assert pid() != b
    with session_scope(seeded) as s:
        assert s.scalar(select(func.count()).select_from(Label)) == 1  # skipping saves nothing

    c = pid()
    button(at, "Exclude photo [X]").click().run()
    with session_scope(seeded) as s:
        assert s.get(Photo, c).excluded and s.get(Photo, c).exclude_reason == "manual"
        assert s.scalar(select(func.count()).select_from(Label)) == 1

    button(at, "Undo [U]").click().run()  # removes the unsure label and shows that photo again
    assert not at.exception, at.exception
    with session_scope(seeded) as s:
        assert s.scalar(select(func.count()).select_from(Label)) == 0
    assert pid() == a


def test_labeling_page_ships_the_keyboard_layer(seeded):
    at = AppTest.from_string(LABELING, default_timeout=60).run()
    assert not at.exception, at.exception
    # every shortcut in the script has a matching on-screen control carrying its [KEY] marker
    labels = (
        " ".join(b.label for b in at.button)
        + " ".join(c.label for c in at.checkbox)
        + " ".join(r.label for r in at.radio)
    )
    for token in ["[1]", "[5]", "[N]", "[S]", "[U]", "[X]", "[W]", "[F]", "[T]", "[C]", "[P]"]:
        assert token in labels, token
    radio_options = " ".join(
        str(o) for r in at.radio if r.key.startswith("room_") for o in r.options
    )
    for token in ["[K]", "[B]", "[L]", "[R]", "[E]", "[O]"]:
        assert token in radio_options, token


def test_training_is_blocked_with_a_clear_message_until_enough_labels(seeded):
    at = AppTest.from_string(LABELING, default_timeout=60).run()
    assert any("Label at least a few dozen" in c.value for c in at.caption)
    assert button(at, "Train from my labels").disabled


# ------------------------------------------------------------------ photo review


def test_photo_review_shows_scores_explanations_and_set_aside_photos(seeded):
    scored(seeded)
    at = AppTest.from_string(REVIEW, default_timeout=60).run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Scored by"] == "Baseline" and metrics["Condition (1 to 5)"] != "n/a"
    assert metrics["Photos used"] == "5"
    shown = [m.value for m in at.markdown]
    assert ":gray-badge[Set aside: floor plan]" in shown
    assert any(v.startswith(":red-badge[Weakest photo in ") for v in shown)
    assert "Limited photo coverage" in " ".join(w.value for w in at.warning)


def test_excluding_a_photo_updates_the_listing_score_immediately(seeded):
    scored(seeded)
    with session_scope(seeded) as s:
        before = s.scalar(select(ListingScore.n_photos_used))
        first = s.scalar(select(Photo).where(Photo.excluded.is_(False)).order_by(Photo.id))
        pid = first.id
    at = AppTest.from_string(REVIEW, default_timeout=60).run()
    at.checkbox(key=f"ex_{pid}").check().run()
    assert not at.exception, at.exception
    with session_scope(seeded) as s:
        assert s.get(Photo, pid).exclude_reason == "manual"
        assert s.scalar(select(ListingScore.n_photos_used)) == before - 1


def test_including_an_auto_excluded_photo_is_remembered_across_rescoring(seeded):
    scored(seeded)
    with session_scope(seeded) as s:
        plan = s.scalar(select(Photo).where(Photo.exclude_reason == "auto: floor plan"))
        pid = plan.id
    at = AppTest.from_string(REVIEW, default_timeout=60).run()
    at.checkbox(key=f"ex_{pid}").uncheck().run()
    with session_scope(seeded) as s:
        assert s.get(Photo, pid).exclude_reason == REASON_KEEP
    scored_again = scored(seeded)  # re-running must respect the user's choice
    with session_scope(seeded) as s:
        assert not s.get(Photo, pid).excluded
    assert scored_again.excluded_junk == 0


def test_score_buttons_are_disabled_note_when_nothing_to_show(settings):
    at = AppTest.from_string(REVIEW, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("No photos yet" in i.value for i in at.info)


# ------------------------------------------------------------------ settings + ranking tie-in


def test_settings_lists_photo_scoring_and_recalculates_on_weight_change(seeded):
    scored(seeded)
    at = AppTest.from_string(SETTINGS, default_timeout=60).run()
    assert not at.exception, at.exception
    at.number_input(key="rw_kitchen").set_value(9.0)
    button(at, "Save room weights").click().run()
    assert not at.exception, at.exception
    assert any("Recalculated" in s.value for s in at.success)
    with session_scope(seeded) as s:
        assert s.scalar(select(ListingScore.weights))["kitchen"] == 9.0


def test_photo_scores_flow_into_the_tier_list_with_weakest_room(seeded):
    from vision.pipeline import refresh_rollups

    scored(seeded)
    with session_scope(seeded) as s:
        for row in s.scalars(select(PhotoScore)):  # pretend the model found a bad kitchen
            row.condition_score = 1.5 if row.room_type == "kitchen" else 4.5
            row.condition_conf = 1.0
        version = s.scalar(select(ListingScore.model_version))
    refresh_rollups(seeded, version)
    res = services.run_ranking(seeded, Preferences(condition_mode="move_in_ready"))
    (r,) = res.ranked
    cond = next(f for f in r.factors if f.key == "condition")
    assert cond.score is not None
    assert "weakest: kitchen" in cond.detail and "limited photo coverage" in cond.detail


def test_keyboard_script_has_no_html_like_text():
    """Streamlit silently skips a script whose body contains angle brackets (found in a real
    browser: a comment mentioning an input tag stopped every shortcut from working)."""
    from app.views.labeling import KEYBOARD_JS

    body = KEYBOARD_JS.strip()
    assert body.startswith("<script>") and body.endswith("</script>")
    inner = body[len("<script>") : -len("</script>")]
    assert "<" not in inner and ">" not in inner.replace("=>", "")
