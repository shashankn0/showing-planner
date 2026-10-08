"""Demo listings and the RentCast source/service."""

import json
import re
from datetime import date

import pytest
import requests
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from app import services
from core.config import Settings
from core.db import init_db, session_scope
from core.models import Listing
from ranking.preferences import Preferences
from sources.demo import DEFAULT_COUNT, demo_listings
from sources.rentcast import RentCastError, RentCastSource, listing_from_payload
from sources.values import parse_date

# A trimmed sample shaped like RentCast's documented sale-listing response (fields per
# https://developers.rentcast.io/reference/sale-listings). Synthetic values.
SAMPLE = [
    {
        "id": "100-Test-St,-Testville,-GA-30301",
        "formattedAddress": "100 Test St, Testville, GA 30301",
        "addressLine1": "100 Test St",
        "addressLine2": "Unit 4B",
        "city": "Testville",
        "state": "GA",
        "zipCode": "30301",
        "county": "Demo",
        "latitude": 33.75,
        "longitude": -84.39,
        "propertyType": "Condo",
        "bedrooms": 2,
        "bathrooms": 2.5,
        "squareFootage": 1100,
        "lotSize": None,
        "yearBuilt": 2005,
        "status": "Active",
        "price": 250000,
        "listedDate": "2026-09-12T00:00:00.000Z",
        "daysOnMarket": 26,
        "mlsNumber": "syn123",
    },
    {"formattedAddress": "200 Sample Ave, Testville, GA 30301", "price": "310000", "bedrooms": 3},
    {"price": 1},  # no address at all
    "not a dict",
]


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeHttp:
    def __init__(self, payload=None, status=200, boom=None):
        self.payload, self.status, self.boom = payload, status, boom
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        if self.boom:
            raise self.boom
        return FakeResponse(self.payload, self.status)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_DATA_DIR", str(tmp_path / "data"))
    for var in ("SHOWING_HOSTED", "APP_PASSWORD", "RENTCAST_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env(load_env_file=False)
    init_db(s)
    return s


# ------------------------------------------------------------------ demo listings


def test_demo_listings_are_deterministic_labeled_and_varied():
    a, b = (
        demo_listings(DEFAULT_COUNT, seed=7, today=date(2026, 10, 8)),
        demo_listings(DEFAULT_COUNT, seed=7, today=date(2026, 10, 8)),
    )
    assert [x.dedupe_key for x in a] == [x.dedupe_key for x in b]
    assert len(a) == DEFAULT_COUNT and len({x.dedupe_key for x in a}) == DEFAULT_COUNT
    assert all(x.is_synthetic and x.source_listing_id.startswith("DEMO") for x in a)
    assert all(x.latitude and x.longitude and x.list_price for x in a)
    assert len({x.property_type for x in a}) >= 2 and len({x.beds for x in a}) >= 3
    assert len({x.status for x in a}) >= 2
    assert all(15 <= x.latitude <= 72 for x in a)


def test_demo_listings_load_once_and_rank(settings):
    first = services.load_demo_listings(settings)
    again = services.load_demo_listings(settings)
    assert first.new == DEFAULT_COUNT and again.new == 0 and again.updated == DEFAULT_COUNT
    res = services.run_ranking(settings, Preferences(max_price=500_000, min_beds=3))
    assert res.ranked and all(r.candidate.is_synthetic for r in res.ranked + res.excluded)
    df = services.listings_table(settings)
    assert df["Demo data"].all() and (df["Map"] == "import").all()


def test_demo_button_adds_listings_on_both_pages(settings):
    page = "from app.views.find_homes import render\nrender()"
    at = AppTest.from_string(page, default_timeout=30).run()
    assert not at.exception, at.exception
    next(b for b in at.button if "demo listings" in b.label).click().run()
    assert not at.exception, at.exception
    assert len(services.listings_table(settings)) == DEFAULT_COUNT
    assert any(s.value.startswith("Tier ") or "Best" in s.value for s in at.subheader) or at.button


# ------------------------------------------------------------------ RentCast mapping


def test_payload_maps_to_listing():
    raw = listing_from_payload(SAMPLE[0])
    assert raw.street == "100 Test St" and raw.unit == "4B"  # "Unit 4B" -> "4B"
    assert (raw.city, raw.state, raw.postal_code) == ("Testville", "GA", "30301")
    assert raw.source == "rentcast" and raw.source_listing_id == "SYN123"
    assert raw.list_price == 250_000 and raw.beds == 2 and raw.baths_total == 2.5
    assert (raw.baths_full, raw.baths_half) == (2, 1)
    assert raw.sqft == 1100 and raw.year_built == 2005 and raw.status == "active"
    assert raw.list_date == date(2026, 9, 12)
    assert (raw.latitude, raw.longitude) == (33.75, -84.39)


def test_payload_without_street_field_falls_back_to_formatted_address():
    raw = listing_from_payload(SAMPLE[1])
    assert raw.street == "200 Sample Ave" and raw.list_price == 310_000 and raw.beds == 3
    assert listing_from_payload(SAMPLE[2]) is None


def test_iso_timestamps_parse():
    assert parse_date("2026-09-12T00:00:00.000Z") == date(2026, 9, 12)
    assert parse_date("2026-09-12 14:30:00") == date(2026, 9, 12)
    assert parse_date("2026-13-45") is None


# ------------------------------------------------------------------ RentCast requests


def test_request_params_and_headers_never_leak_the_key_into_params():
    http = FakeHttp(SAMPLE)
    src = RentCastSource("SECRET-KEY", city="Testville", state="ga", max_price=500_000,
                         property_types=["Condo"], limit=9999, http=http)  # fmt: skip
    params = src.request_params()
    assert params == {"status": "Active", "limit": 500, "city": "Testville", "state": "GA",
                      "price": "0-500000", "propertyType": "Condo"}  # fmt: skip
    assert "SECRET-KEY" not in json.dumps(params)
    got = list(src.fetch())
    url, kw = http.calls[0]
    assert url == "https://api.rentcast.io/v1/listings/sale"
    assert kw["headers"]["X-Api-Key"] == "SECRET-KEY"
    assert [g.street for g in got] == ["100 Test St", "200 Sample Ave"] and src.skipped == 2


def test_zip_search_and_multi_type_filtering():
    src = RentCastSource(
        "k", zip_code="30301-1234", property_types=["Condo", "Townhouse"], http=FakeHttp(SAMPLE)
    )
    assert src.request_params()["zipCode"] == "30301" and "propertyType" not in src.request_params()
    got = list(src.fetch())
    assert [g.street for g in got] == ["100 Test St"]  # the type-less result is filtered out


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "ZIP code, or a city"),
        ({"city": "Testville"}, "ZIP code, or a city"),
    ],
)
def test_location_is_required_before_any_call(kwargs, message):
    with pytest.raises(RentCastError, match=message):
        RentCastSource("k", http=FakeHttp(SAMPLE), **kwargs).request_params()


@pytest.mark.parametrize(
    ("status", "text"),
    [(401, "didn't accept that API key"), (403, "didn't accept that API key"),
     (429, "request limit"), (400, "error (400)"), (500, "error (500)")],
)  # fmt: skip
def test_http_errors_become_plain_language(status, text):
    src = RentCastSource("k", zip_code="30301", http=FakeHttp({"message": "bad"}, status=status))
    with pytest.raises(RentCastError, match=re.escape(text)):
        src.fetch_payload()


def test_network_failure_and_bad_payloads():
    boom = RentCastSource("k", zip_code="30301", http=FakeHttp(boom=requests.ConnectionError("x")))
    with pytest.raises(RentCastError, match="Couldn't reach RentCast"):
        boom.fetch_payload()
    for payload in ({"not": "a list"}, ValueError("not json")):
        src = RentCastSource("k", zip_code="30301", http=FakeHttp(payload))
        with pytest.raises(RentCastError):
            src.fetch_payload()
    with pytest.raises(RentCastError, match="Paste your RentCast API key"):
        RentCastSource("  ", zip_code="30301", http=FakeHttp(SAMPLE)).fetch_payload()


def test_error_messages_never_contain_the_key():
    for status in (401, 429, 500):
        src = RentCastSource(
            "TOP-SECRET", zip_code="30301", http=FakeHttp({"message": "m"}, status=status)
        )
        with pytest.raises(RentCastError) as exc:
            src.fetch_payload()
        assert "TOP-SECRET" not in str(exc.value)


# ------------------------------------------------------------------ RentCast service (cache, privacy)


def test_search_imports_listings_and_caches_without_the_key(settings):
    http = FakeHttp(SAMPLE)
    res = services.rentcast_search(settings, "TOP-SECRET", zip_code="30301", http=http, now=1000.0)
    assert (res.fetched, res.from_cache, res.summary.new, res.skipped) == (4, False, 2, 2)
    assert len(http.calls) == 1
    # repeating the search spends no more quota
    res2 = services.rentcast_search(
        settings, "TOP-SECRET", zip_code="30301", http=http, now=1000.0 + 600
    )
    assert res2.from_cache and res2.cache_age_minutes == 10 and len(http.calls) == 1
    assert res2.summary.new == 0 and res2.summary.updated == 2
    # forcing a refresh, or waiting past the cache window, calls again
    services.rentcast_search(
        settings, "TOP-SECRET", zip_code="30301", http=http, refresh=True, now=1700.0
    )
    services.rentcast_search(settings, "TOP-SECRET", zip_code="30301", http=http,
                             now=1700.0 + services.RENTCAST_CACHE_SECONDS + 5)  # fmt: skip
    assert len(http.calls) == 3
    # the key must not be on disk anywhere: cache files or the database
    for f in settings.data_dir.rglob("*"):
        if f.is_file():
            assert b"TOP-SECRET" not in f.read_bytes(), f"key found in {f}"


def test_different_searches_do_not_share_a_cache(settings):
    http = FakeHttp(SAMPLE)
    services.rentcast_search(settings, "k", zip_code="30301", http=http, now=1.0)
    services.rentcast_search(settings, "k", zip_code="30302", http=http, now=2.0)
    assert len(http.calls) == 2


def test_failed_search_spends_nothing_and_imports_nothing(settings):
    with pytest.raises(RentCastError):
        services.rentcast_search(settings, "k", zip_code="30301", http=FakeHttp(status=401))
    with pytest.raises(RentCastError, match="ZIP code, or a city"):
        services.rentcast_search(settings, "k", http=FakeHttp(SAMPLE))
    with session_scope(settings) as s:
        assert not list(s.scalars(select(Listing)))
    assert not (settings.data_dir / "cache").exists() or not list(
        (settings.data_dir / "cache").iterdir()
    )


def test_search_results_flow_into_ranking(settings):
    services.rentcast_search(settings, "k", zip_code="30301", http=FakeHttp(SAMPLE), now=1.0)
    res = services.run_ranking(settings, Preferences(max_price=300_000, min_beds=2))
    assert [r.candidate.address.split(",")[0] for r in res.ranked] == ["100 Test St #4B"]


# ------------------------------------------------------------------ the page


PAGE = "from app.views.listings import render\nrender()"


def _spy_on_search(monkeypatch):
    """Record the API key the page passes to the service, without any network access."""
    seen = {}

    def spy(settings, api_key, **kw):
        seen["key"] = api_key
        raise RentCastError("stopped by test")  # never reaches RentCast

    monkeypatch.setattr(services, "rentcast_search", spy)
    return seen


def test_hosted_page_never_falls_back_to_the_owners_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_HOSTED", "1")
    monkeypatch.setenv("RENTCAST_API_KEY", "OWNERS-KEY")
    from app import ui

    monkeypatch.setattr(ui, "SESSION_ROOT", tmp_path / "sessions")
    assert ui.load_settings().rentcast_api_key == "OWNERS-KEY"  # it is in the environment...
    seen = _spy_on_search(monkeypatch)
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    assert not at.exception, at.exception
    at.text_input(key="rc_zip").set_value("30301")
    next(b for b in at.button if "Search and add" in b.label).click().run()
    assert seen["key"] == ""  # ...but a visitor who typed nothing gets no key at all


def test_hosted_page_uses_the_key_the_visitor_typed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOWING_HOSTED", "1")
    from app import ui

    monkeypatch.setattr(ui, "SESSION_ROOT", tmp_path / "sessions")
    seen = _spy_on_search(monkeypatch)
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    at.text_input[0].set_value("  visitors-own-key  ")
    at.text_input(key="rc_zip").set_value("30301")
    next(b for b in at.button if "Search and add" in b.label).click().run()
    assert seen["key"] == "visitors-own-key"


def test_private_mode_page_uses_the_env_key_when_none_typed(settings, monkeypatch):
    monkeypatch.setenv("RENTCAST_API_KEY", "MY-ENV-KEY")
    seen = _spy_on_search(monkeypatch)
    at = AppTest.from_string(PAGE, default_timeout=30).run()
    at.text_input(key="rc_zip").set_value("30301")
    next(b for b in at.button if "Search and add" in b.label).click().run()
    assert seen["key"] == "MY-ENV-KEY"
    assert any("stopped by test" in e.value for e in at.error)  # errors surface as messages
