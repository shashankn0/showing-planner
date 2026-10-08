"""Find Homes page, client profiles, destination lookup, and hosted-mode protections."""

import os
import time
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from app import services, ui
from core.config import Settings
from core.db import init_db, session_scope
from core.models import Listing, ListingScore
from ranking.preferences import Preferences
from sources.csv_import import analyze_file

FIX = Path(__file__).parent / "fixtures"
MAIN = str(Path(__file__).parents[1] / "app" / "main.py")
PAGE = "from app.views.find_homes import render\nrender()"


class FakeCensus:
    def __init__(self, match=True):
        self.match = match

    def post(self, url, **kw):
        import csv

        rows = []
        for line in csv.reader(kw["files"]["addressFile"][1].strip().splitlines()):
            if self.match:
                rows.append(
                    f'"{line[0]}","{line[1]}","Match","Exact","{line[1]}","-84.39,33.75","1","L"'
                )
            else:
                rows.append(f'"{line[0]}","{line[1]}","No_Match"')

        class R:
            text = "\n".join(rows)

            def raise_for_status(self):
                pass

        return R()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SHOWING_HOSTED", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    return s


@pytest.fixture
def seeded(settings):
    a = analyze_file(FIX / "synthetic_portal_style.csv")
    services.run_import(settings, a, "p.csv", is_synthetic=True, geocode=False)
    return settings


# ------------------------------------------------------------------ services


def test_candidates_pick_up_photo_condition_when_scored(seeded):
    with session_scope(seeded) as s:
        lst = s.scalar(select(Listing).where(Listing.source_listing_id == "SYN9200001"))
        s.add(ListingScore(listing_id=lst.id, model_version="t", overall=4.2, n_photos_used=7))
    cands = {c.mls: c for c in services.candidates_from_db(seeded)}
    assert cands["SYN9200001"].condition == 4.2 and cands["SYN9200001"].condition_photos == 7
    assert cands["SYN9200002"].condition is None
    assert all(c.is_synthetic for c in cands.values())


def test_end_to_end_ranking_from_database(seeded):
    prefs = Preferences(max_price=520_000, min_beds=3, min_baths=2, min_sqft=1500, ideal_sqft=2200)
    res = services.run_ranking(seeded, prefs)
    ranked = {r.candidate.mls: r for r in res.ranked}
    assert set(ranked) == {"SYN9200001", "SYN9200002", "SYN9200004"}  # the 1-bed condo is out
    assert [r.candidate.mls for r in res.excluded] == ["SYN9200003"]
    df = services.ranking_table(res)
    assert list(df["Tier"]) == [r.tier for r in res.ranked]
    assert "Strengths" in df.columns


def test_client_profiles_round_trip(seeded):
    prefs = Preferences(max_price=500_000, min_beds=3, condition_mode="move_in_ready")
    cid = services.save_client(seeded, "  Pat & Sam  ", prefs)
    assert services.list_clients(seeded) == [(cid, "Pat & Sam")]
    again = services.load_client_prefs(seeded, cid)
    assert again == prefs
    # saving the same nickname updates instead of duplicating
    prefs.max_price = 450_000
    assert services.save_client(seeded, "Pat & Sam", prefs) == cid
    assert services.load_client_prefs(seeded, cid).max_price == 450_000
    with pytest.raises(ValueError, match="nickname"):
        services.save_client(seeded, "   ", prefs)


def test_destination_lookup_success_and_failure(settings):
    dest, msg = services.resolve_destination(
        settings, "Work", "100 Main St, Testville, GA 30301", max_minutes=30, http=FakeCensus()
    )
    assert msg == "" and dest.name == "Work" and dest.max_minutes == 30
    assert dest.lat == pytest.approx(33.75) and dest.address.startswith("100 Main St")
    # a different address that the (fake) service can't match -> clear message, no destination
    none, msg = services.resolve_destination(
        settings, "Work", "999 Nowhere Ave, Testville, GA 30301", http=FakeCensus(match=False)
    )
    assert none is None and "Couldn't find" in msg
    none, msg = services.resolve_destination(settings, "Work", "hello", http=FakeCensus())
    assert none is None


# ------------------------------------------------------------------ page


def test_find_homes_shows_empty_state_without_listings(settings):
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("No listings to rank" in i.value for i in at.info)


def test_find_homes_builds_a_tier_list(seeded):
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    assert not at.exception, at.exception
    at.number_input(key="c0_maxp").set_value(520_000)
    at.number_input(key="c0_beds").set_value(3)
    at.number_input(key="c0_baths").set_value(2.0)
    at.number_input(key="c0_sqft").set_value(1500)
    at.text_input(key="c0_save").set_value("Test Client")
    at.button[0].click().run()
    assert not at.exception, at.exception
    subheaders = [s.value for s in at.subheader]
    assert any(s.startswith("Tier ") for s in subheaders)
    assert any("Not considered" in e.label for e in at.expander)
    assert services.list_clients(seeded)[0][1] == "Test Client"
    assert any("Pretend Pl" in m.value for m in at.markdown)


def test_find_homes_explains_when_nothing_matches(seeded):
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    at.number_input(key="c0_maxp").set_value(50_000)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("meet the must-haves" in w.value for w in at.warning)


def test_main_app_lists_find_homes_first(settings):
    at = AppTest.from_file(MAIN, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("Find Homes" in t.value for t in at.title)


# ------------------------------------------------------------------ hosted mode


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_HOSTED", "1")
    monkeypatch.setattr(ui, "SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(ui, "_swept", False)
    return tmp_path / "sessions"


def test_hosted_visitors_cannot_see_each_others_data(hosted):
    one = AppTest.from_string(PAGE, default_timeout=30).run()
    ws1 = Path(one.session_state["_workspace"])
    s1 = Settings(data_dir=ws1)
    services.add_addresses(s1, "12 Sample Rd, Testville, GA 30301", geocode=False)

    two = AppTest.from_string(PAGE, default_timeout=30).run()
    ws2 = Path(two.session_state["_workspace"])
    assert ws1 != ws2 and ws1.parent == hosted
    assert any("No listings to rank" in i.value for i in two.info)
    assert services.listings_table(Settings(data_dir=ws2)).empty
    assert len(services.listings_table(s1)) == 1
    assert any("Shared website mode" in i.value for i in two.info)

    one.run()  # the first visitor still has their data after a rerun
    assert not any("No listings to rank" in i.value for i in one.info)
    assert one.session_state["_workspace"] == str(ws1)


def test_stale_workspaces_are_swept(hosted):
    old = hosted / "s_old"
    old.mkdir(parents=True)
    ancient = time.time() - 7 * 3600
    os.utime(old, (ancient, ancient))
    fresh = hosted / "s_fresh"
    fresh.mkdir()
    AppTest.from_string(PAGE, default_timeout=30).run()
    assert not old.exists() and fresh.exists()


def test_hosted_listings_page_offers_only_zip_upload(hosted):
    at = AppTest.from_string(
        "from app.views.listings import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception
    labels = [r.label for r in at.radio]
    assert not any("Where are the photos" in label for label in labels)
    assert not any("Keep the photos" in label for label in labels)


# ------------------------------------------------------------------ password gate


def test_password_gate_blocks_until_correct(settings, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "open-sesame")
    at = AppTest.from_file(MAIN, default_timeout=30).run()
    assert not at.exception, at.exception
    assert not any("Find Homes" in t.value for t in at.title)
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert any("isn't right" in e.value for e in at.error)
    assert not any("Find Homes" in t.value for t in at.title)
    at.text_input[0].set_value("open-sesame")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Find Homes" in t.value for t in at.title)


def test_password_gate_locks_after_too_many_attempts(settings, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "open-sesame")
    at = AppTest.from_file(MAIN, default_timeout=30).run()
    for _ in range(ui.MAX_PASSWORD_ATTEMPTS):
        at.text_input[0].set_value("bad")
        at.button[0].click().run()
    assert any("Too many attempts" in e.value for e in at.error)


# ------------------------------------------------------------------ exposure detection


@pytest.mark.parametrize(
    ("host", "exposed"),
    [
        ("localhost:8501", False),
        ("127.0.0.1:8501", False),
        ("192.168.1.20:8501", False),
        ("10.0.0.5:8501", False),
        ("172.20.1.1:8501", False),
        ("172.40.1.1:8501", True),
        ("[::1]:8501", False),
        ("MyLaptop.local:8501", False),
        ("random-words.trycloudflare.com", True),
        ("my-app.streamlit.app", True),
        ("203.0.113.9:8501", True),
        (None, False),
    ],
)
def test_exposed_host_detection(host, exposed):
    assert ui.is_exposed_host(host) is exposed


def test_secrets_can_turn_on_hosted_mode(settings, monkeypatch):
    monkeypatch.setattr(ui.st, "secrets", {"SHOWING_HOSTED": "true", "APP_PASSWORD": "pw"})
    s = ui.load_settings()
    assert s.hosted and s.app_password == "pw"
