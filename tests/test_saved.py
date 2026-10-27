"""Saved homes (tracking), the Google helpers, and the optional sign-in."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from app import earth, services
from core.config import Settings
from core.db import init_db, session_scope
from core.models import Listing, SavedHome

ROOT = Path(__file__).parents[1]
KML = "{http://www.opengis.net/kml/2.2}"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    services.add_addresses(
        s, "1 Test Rd, Testville, GA 30301\n2 Test Rd, Testville, GA 30301\n", geocode=False
    )
    return s


def ids(settings):
    with session_scope(settings) as s:
        return [x.id for x in s.scalars(select(Listing).order_by(Listing.id))]


# ------------------------------------------------------------------ tracking


def test_save_is_idempotent_and_defaults_to_interested(settings):
    a, _ = ids(settings)
    services.save_home(settings, a)
    services.save_home(settings, a)
    (home,) = services.saved_homes(settings)
    assert (home.listing_id, home.status, home.notes) == (a, "interested", "")
    assert services.saved_ids(settings) == {a}


def test_status_and_notes_update_and_bad_status_is_refused(settings):
    a, _ = ids(settings)
    services.save_home(settings, a)
    services.update_saved(settings, a, status="toured", notes="  likes the yard  ")
    (home,) = services.saved_homes(settings)
    assert (home.status, home.notes) == ("toured", "likes the yard")
    services.update_saved(settings, a, notes="")
    assert services.saved_homes(settings)[0].notes == ""
    with pytest.raises(ValueError, match="Status must be"):
        services.update_saved(settings, a, status="sold")
    with pytest.raises(ValueError, match="isn't saved"):
        services.update_saved(settings, ids(settings)[1], status="offer")


def test_unsave_and_newest_first(settings):
    a, b = ids(settings)
    services.save_home(settings, a)
    services.save_home(settings, b)
    assert [h.listing_id for h in services.saved_homes(settings)] == [b, a]
    services.unsave_home(settings, b)
    services.unsave_home(settings, b)  # already gone: no error
    assert services.saved_ids(settings) == {a}


def test_save_refuses_a_listing_that_does_not_exist(settings):
    with pytest.raises(ValueError, match="no longer exists"):
        services.save_home(settings, 9999)


def test_deleting_a_listing_removes_its_saved_row(settings):
    a, _ = ids(settings)
    services.save_home(settings, a)
    with session_scope(settings) as s:
        s.delete(s.get(Listing, a))
    with session_scope(settings) as s:
        assert s.scalar(select(SavedHome)) is None


# ------------------------------------------------------------------ google helpers


def item(**kw):
    base = dict(
        listing_id=1, address="1 A St", price=450000, beds=3, baths=2.0, sqft=1800,
        lat=33.749, lon=-84.388, status="toured", notes="private note", is_synthetic=False,
    )  # fmt: skip
    return services.SavedItem(**{**base, **kw})


def test_kml_is_valid_skips_unplaced_homes_and_leaves_notes_out():
    out = earth.kml([item(address="Fish & Chips <Rd> \x07"), item(listing_id=2, lat=None)])
    root = ET.fromstring(out)
    marks = root.findall(f".//{KML}Placemark")
    assert len(marks) == 1
    assert marks[0].find(f"{KML}name").text == "Fish & Chips <Rd> "
    assert marks[0].find(f"{KML}description").text == "toured · $450,000"
    assert marks[0].find(f".//{KML}coordinates").text == "-84.388000,33.749000,0"  # lon,lat
    assert "private note" not in out


def test_urls_are_encoded_and_need_no_key():
    assert earth.maps_url("1 A St & B, Atlanta") == (
        "https://www.google.com/maps/search/?api=1&query=1+A+St+%26+B%2C+Atlanta"
    )
    sat = earth.satellite_url(33.749, -84.388)
    assert "map_action=map" in sat and "basemap=satellite" in sat
    assert "center=33.749%2C-84.388" in sat and "key=" not in sat


def test_viewer_html_carries_the_key_safely_and_the_elevation_fallback():
    html = earth.viewer_html(33.749, -84.388, 'a"b</script>')
    assert "key=a%22b%3C%2Fscript%3E&" in html  # nothing can break out of the URL
    assert "</script>&" not in html
    assert 'center="33.749000,-84.388000,0"' in html  # still opens when the lookup fails
    assert "x=-84.388000&y=33.749000" in html and "libraries=maps3d" in html
    with pytest.raises(ValueError):
        earth.viewer_html("33.7; alert(1)", -84.388, "k")  # coordinates are numbers or an error


# ------------------------------------------------------------------ pages


def test_saved_page_shows_homes_and_the_viewer(settings):
    a, b = ids(settings)
    services.save_home(settings, a)
    services.update_saved(settings, a, status="offer")
    at = AppTest.from_string(
        "from app.views.saved import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric] == ["0", "0", "1", "0"]
    assert at.selectbox(key="view_home").value == a


def test_save_button_on_a_tier_card_saves_the_home(settings):
    services.load_demo_listings(settings)
    at = AppTest.from_string(
        "from app.views.find_homes import render\nrender()", default_timeout=60
    ).run()
    next(b for b in at.button if "Build tier list" in b.label).click().run()
    assert not at.exception, at.exception
    save = next(b for b in at.button if b.label == "☆ Save")
    listing_id = int(save.key.removeprefix("save_"))
    save.click().run()
    assert not at.exception, at.exception
    assert services.saved_ids(settings) == {listing_id}
    assert f"save_{listing_id}" not in {b.key for b in at.button}  # now shows ★ Saved


# ------------------------------------------------------------------ sign-in


def test_nothing_changes_when_sign_in_is_not_configured(settings):
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert not any("Sign in" in b.label for b in at.button)


def test_signed_out_visitors_see_only_the_sign_in_button(settings, monkeypatch):
    monkeypatch.setattr("app.ui.auth_configured", lambda: True)
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert [b.label for b in at.button] == ["Sign in with Google"]
    assert not at.sidebar.caption  # no app navigation or data behind the gate


def test_each_signed_in_user_gets_their_own_data_folder(settings, monkeypatch):
    monkeypatch.setattr("app.ui.signed_in_email", lambda: "ann@example.com")
    at = AppTest.from_string(
        "from app.ui import context\nsettings = context()", default_timeout=30
    ).run()
    assert not at.exception, at.exception
    who = hashlib.sha256(b"ann@example.com").hexdigest()[:16]
    assert at.session_state["_db_ready"] == str(settings.data_dir / "users" / who)
    assert (settings.data_dir / "users" / who / "app.db").exists()


def test_settings_page_explains_sign_in_when_it_is_off(settings):
    at = AppTest.from_string(
        "from app.views.settings import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception
    assert "Google sign-in" in [h.value for h in at.subheader]
    assert any("[auth]" in c.value for c in at.code)  # the copy-paste template is shown


def test_settings_page_shows_who_is_signed_in(settings, monkeypatch):
    monkeypatch.setattr("app.views.settings.auth_configured", lambda: True)
    monkeypatch.setattr("app.views.settings.signed_in_email", lambda: "ann@example.com")
    at = AppTest.from_string(
        "from app.views.settings import render\nrender()", default_timeout=30
    ).run()
    assert not at.exception, at.exception
    assert any("ann@example.com" in s.value for s in at.success)
    assert not any("[auth]" in c.value for c in at.code)  # no template once it's set up


def test_secrets_template_has_every_key_streamlit_needs():
    import tomllib

    from app.views.settings import AUTH_EXAMPLE

    auth = tomllib.loads(AUTH_EXAMPLE.read_text(encoding="utf-8"))["auth"]
    assert {"redirect_uri", "cookie_secret", "client_id", "client_secret"} <= set(auth)
    assert auth["server_metadata_url"].endswith("/.well-known/openid-configuration")
