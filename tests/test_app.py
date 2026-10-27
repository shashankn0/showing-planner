"""Service-level and page-render tests for the Streamlit app (no browser needed)."""

from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from app import services
from core.config import Settings
from core.db import init_db, session_scope
from core.models import Listing, Photo
from sources.csv_import import analyze_file

FIX = Path(__file__).parent / "fixtures"
PAGES = ["find_homes", "saved", "listings", "photo_review", "labeling", "plan_day", "settings"]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    return s


class FakeCensus:
    """Minimal Census batch stand-in: matches everything near Atlanta."""

    def post(self, url, **kw):
        import csv

        body = kw["files"]["addressFile"][1]
        out = []
        for line in csv.reader(body.strip().splitlines()):
            out.append(
                f'"{line[0]}","{line[1]}","Match","Exact","{line[1]}","-84.39,33.75","1","L"'
            )

        class R:
            text = "\n".join(out)

            def raise_for_status(self):
                pass

        return R()


def make_img(path: Path):
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    r, g, b = hashlib.md5(str(path).encode()).digest()[:3]  # distinct pixels per file
    Image.new("RGB", (80, 60), (r, g, b)).save(path)


# ------------------------------------------------------------------ services


def test_import_flow_end_to_end(settings):
    a = analyze_file(FIX / "synthetic_mls_style.csv", profiles_dir=settings.mappings_dir)
    preview = services.preview_rows(a, settings)
    assert len(preview) == 6 and "Address" in preview.columns
    summary = services.run_import(
        settings, a, "synthetic_mls_style.csv", is_synthetic=True,
        save_layout_as="MLS style", geocode=True, http=FakeCensus(),
    )  # fmt: skip
    assert (summary.new, summary.updated) == (6, 0)
    assert summary.geocode.ok == 6
    again = analyze_file(FIX / "synthetic_mls_style.csv", profiles_dir=settings.mappings_dir)
    assert again.profile_name == "MLS style"
    df = services.listings_table(settings)
    assert len(df) == 6 and df["Demo data"].all() and (df["Map"] == "ok").all()


def test_import_refuses_when_no_address(settings):
    a = analyze_file(b"Color,Shape\nred,square\nblue,round\n", "x.csv")
    with pytest.raises(ValueError, match="No address"):
        services.run_import(settings, a, "x.csv")


def test_add_addresses_and_skip_unreadable_lines(settings):
    s = services.add_addresses(
        settings, "12 Sample Rd, Testville, GA 30301\nhello\n", geocode=False
    )
    assert s.new == 1 and len(s.skipped) == 1


def test_photo_flow_with_matching_unmatched_and_new_listing(settings, tmp_path):
    services.add_addresses(settings, "1001 Synthetic Way, Testville, GA 30301\n", geocode=False)
    root = tmp_path / "photos_in"
    make_img(root / "1001 Synthetic Way" / "a.jpg")
    make_img(root / "1001 Synthetic Way" / "b.jpg")
    make_img(root / "77 Brand New Ln" / "a.jpg")
    make_img(root / "camera_dump" / "x.jpg")
    scan = services.scan_photos(settings, folder=root)
    assert len(scan.report.matches) == 1
    assert scan.new_listing_keys == {"77 Brand New Ln"}
    assignments = {m.group.key: m.listing_id for m in scan.report.matches}
    assignments["77 Brand New Ln"] = services.NEW_LISTING
    assignments["camera_dump"] = None
    res = services.apply_photos(settings, scan, assignments)
    assert res.added == 3
    with session_scope(settings) as s:
        assert len(list(s.scalars(select(Listing)))) == 2
        assert len(list(s.scalars(select(Photo)))) == 3


def test_scan_rejects_a_folder_that_does_not_exist(settings, tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        services.scan_photos(settings, folder=tmp_path / "nope")


# ------------------------------------------------------------------ pages render


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_an_empty_database(settings, page):
    at = AppTest.from_string(
        f"from app.views.{page} import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_data(settings, page, tmp_path):
    services.add_addresses(settings, "1001 Synthetic Way, Testville, GA 30301\n", geocode=False)
    with session_scope(settings) as s:
        lst = s.scalar(select(Listing))
    src = tmp_path / "p" / "a.jpg"
    make_img(src)
    from sources.photos import ingest_photos

    with session_scope(settings) as s:
        ingest_photos(s, settings, s.get(Listing, lst.id), [src])
    at = AppTest.from_string(
        f"from app.views.{page} import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception


def test_main_app_boots_with_navigation(settings):
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert any("Find Homes" in t.value for t in at.title)  # landing page


def test_settings_page_saves_defaults(settings):
    at = AppTest.from_string(
        "from app.views.settings import render\nrender()", default_timeout=30
    ).run()
    at.text_input[0].set_value("Testville")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert Settings.from_env(load_env_file=False).prefs.default_city == "Testville"
