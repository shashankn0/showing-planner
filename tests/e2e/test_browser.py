"""Real-browser check: a live Streamlit server driven by headless Chromium.

Opt-in because it needs Playwright and a browser download:
    pip install playwright && python -m playwright install chromium
    E2E=1 pytest tests/e2e -q            (screenshots land in <tmp>/shots and are printed)

It proves what the in-process tests can't: that the keyboard shortcuts really work in a browser
and that every page renders without an error box.
"""

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select

pytestmark = pytest.mark.skipif(not os.environ.get("E2E"), reason="set E2E=1 to run browser tests")
sync_api = pytest.importorskip("playwright.sync_api")

from core.config import Settings  # noqa: E402
from core.db import init_db, session_scope  # noqa: E402
from core.models import Label, Listing, Photo  # noqa: E402
from core.store import upsert_listings  # noqa: E402
from sources.base import RawListing  # noqa: E402
from sources.demo import demo_listings  # noqa: E402
from sources.photos import ingest_photos  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_photo(seed: int, path: Path) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, 3)
    blocks = rng.normal(0, 25, (6, 8, 1))
    pattern = np.kron(blocks, np.ones((60, 80, 1)))[:320, :420]
    arr = np.clip(base + pattern + rng.normal(0, 6, (320, 420, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=90)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data = tmp / "data"
    settings = Settings(data_dir=data)
    init_db(settings)
    with session_scope(settings) as s:
        upsert_listings(s, demo_listings(30), None)
        upsert_listings(
            s,
            [
                RawListing(
                    "manual", "1 Browser St", "Testville", "GA", "30301", source_listing_id="E2E1"
                )
            ],
        )
        listing = s.scalar(select(Listing).where(Listing.source_listing_id == "E2E1"))
        files = []
        for i in range(12):
            p = tmp / f"e2e_{i}.jpg"
            make_photo(100 + i, p)
            files.append(p)
        ingest_photos(s, settings, listing, files)
    port = free_port()
    env = {**os.environ, "SHOWING_DATA_DIR": str(data), "PYTHONIOENCODING": "utf-8"}
    for k in ("SHOWING_HOSTED", "APP_PASSWORD"):
        env.pop(k, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.headless", "true",
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=2).status == 200:
                break
        except OSError:
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail("Streamlit did not start")
    yield {"url": url, "settings": settings, "shots": tmp / "shots"}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def page(app):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium isn't installed: {exc}")
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
        pg = ctx.new_page()
        app["shots"].mkdir(exist_ok=True)
        yield pg
        browser.close()


def shot(app, page, name):
    path = app["shots"] / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"screenshot: {path}")


def wait_for(fn, seconds=8):
    end = time.time() + seconds
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.25)
    return False


def labels(app):
    with session_scope(app["settings"]) as s:
        return [
            (lab.condition, lab.room_type, tuple(lab.flags or []), lab.unsure)
            for lab in s.scalars(select(Label).order_by(Label.id))
        ]


def open_labeling(app, page):
    page.goto(app["url"] + "/labeling")
    page.get_by_text("Condition", exact=False).first.wait_for(timeout=30000)
    page.get_by_role("button", name="3 [3]").wait_for(timeout=30000)
    page.wait_for_timeout(800)  # let the keyboard script attach


def test_labeling_keyboard_shortcuts_work_in_a_real_browser(app, page):
    open_labeling(app, page)
    shot(app, page, "labeling_start")
    assert labels(app) == []

    page.keyboard.press("3")  # rate 3
    assert wait_for(lambda: len(labels(app)) == 1), "pressing 3 did not save a label"
    assert labels(app)[0][0] == 3

    page.wait_for_timeout(1000)
    page.keyboard.press("b")  # room: bathroom
    page.wait_for_timeout(800)
    page.keyboard.press("w")  # flag: water stain
    page.wait_for_timeout(500)
    page.keyboard.press("c")  # flag: clutter
    page.wait_for_timeout(500)
    shot(app, page, "labeling_before_save")
    page.keyboard.press("2")  # rate 2 -> saves with room + flags
    assert wait_for(lambda: len(labels(app)) == 2), "pressing 2 did not save a label"
    cond, room, flags, unsure = labels(app)[1]
    assert (cond, room, unsure) == (2, "bath", False)
    assert set(flags) == {"water_stain", "clutter"}

    page.wait_for_timeout(1000)
    page.keyboard.press("n")  # unsure
    assert wait_for(lambda: len(labels(app)) == 3)
    assert labels(app)[2][3] is True and labels(app)[2][0] is None

    page.wait_for_timeout(1000)
    page.keyboard.press("u")  # undo the unsure label
    assert wait_for(lambda: len(labels(app)) == 2), "undo did not remove the last label"

    page.wait_for_timeout(1000)
    page.keyboard.press("s")  # skip saves nothing
    page.wait_for_timeout(1500)
    assert len(labels(app)) == 2

    page.wait_for_timeout(1000)
    page.keyboard.press("x")  # exclude photo
    assert wait_for(lambda: _excluded_count(app) == 1), "exclude did not mark a photo"
    shot(app, page, "labeling_after")


def _excluded_count(app) -> int:
    with session_scope(app["settings"]) as s:
        return s.scalar(select(func.count()).select_from(Photo).where(Photo.excluded.is_(True)))


def test_typing_in_a_text_box_does_not_trigger_shortcuts(app, page):
    page.goto(app["url"] + "/settings")
    box = page.get_by_label("Default city")
    box.wait_for(timeout=30000)
    before = len(labels(app))
    box.click()
    box.type("12345 skunk")  # contains digits and shortcut letters
    page.wait_for_timeout(800)
    assert len(labels(app)) == before


@pytest.mark.parametrize(
    "path", ["", "listings", "photo-review", "labeling", "plan-a-day", "settings"]
)
def test_every_page_renders_in_a_browser_without_an_error_box(app, page, path):
    page.goto(f"{app['url']}/{path}")
    page.locator("h1").first.wait_for(timeout=30000)
    page.wait_for_timeout(1500)
    assert page.locator('[data-testid="stException"]').count() == 0
    # Streamlit shows a modal "Page not found" if a route is wrong (it still renders the home page behind it)
    assert page.locator('[data-testid="stDialog"]').count() == 0, page.locator(
        '[data-testid="stDialog"]'
    ).first.inner_text()
    shot(app, page, (path or "home").replace("-", "_"))


def test_find_homes_builds_a_tier_list_in_a_browser(app, page):
    page.goto(app["url"] + "/")
    page.get_by_role("button", name="Build tier list").wait_for(timeout=30000)
    page.get_by_label("Maximum price ($)").fill("550000")
    page.get_by_label("Maximum price ($)").press("Enter")
    page.get_by_label("Bedrooms (at least)").fill("3")
    page.get_by_label("Bedrooms (at least)").press("Enter")
    page.wait_for_timeout(500)
    page.get_by_role("button", name="Build tier list").click()
    page.get_by_text("Tier ", exact=False).first.wait_for(timeout=30000)
    assert page.locator('[data-testid="stException"]').count() == 0
    # wait for Streamlit to finish rerunning, so we don't inspect a half-updated page
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=30000
    )
    page.wait_for_timeout(500)
    tiles = page.locator('[data-testid="stMetric"]').filter(has_text=re.compile(r"^Tier ")).count()
    groups = page.get_by_role("heading", name=re.compile(r"^Tier [SABCD?] ")).count()
    assert tiles == groups > 0, f"{tiles} summary tiles for {groups} tier groups"
    assert page.locator(".katex").count() == 0, "a price line was rendered as a math formula"
    assert (
        "at or under the $" in page.locator("body").inner_text()
        or "over the $" in page.locator("body").inner_text()
    )
    shot(app, page, "find_homes_results")
