import json

import pytest
import requests
from sqlalchemy import func, select

from core.config import DEFAULT_NOMINATIM_UA, Settings
from core.db import init_db, session_scope
from core.geocoding import Geocoder
from core.models import GeocodeCache, Listing
from core.store import upsert_listings
from sources.base import RawListing


class FakeResp:
    def __init__(self, text="", payload=None, status=200):
        self.text, self._payload, self.status_code = text, payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


class FakeHttp:
    """Stands in for requests.Session. Census answers by address text; Nominatim by street."""

    def __init__(self, census=None, nominatim=None, fail=False):
        self.census, self.nominatim, self.fail = census or {}, nominatim or {}, fail
        self.posts, self.gets = [], []

    def post(self, url, **kw):
        self.posts.append(kw)
        if self.fail:
            raise requests.ConnectionError("offline")
        body = kw["files"]["addressFile"][1]
        lines = []
        for line in body.strip().splitlines():
            idx, street, *_ = next(__import__("csv").reader([line]))
            hit = self.census.get(street)
            if hit:
                lines.append(
                    f'"{idx}","{street}","Match","Exact","{street} MATCHED","{hit[1]},{hit[0]}","1","L"'
                )
            else:
                lines.append(f'"{idx}","{street}","No_Match"')
        return FakeResp("\n".join(lines))

    def get(self, url, **kw):
        self.gets.append(kw)
        hit = self.nominatim.get(kw["params"]["street"])
        return FakeResp(
            payload=[{"lat": str(hit[0]), "lon": str(hit[1]), "display_name": "x"}] if hit else []
        )


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data", nominatim_user_agent="tests/1.0 (qa@example.invalid)")
    init_db(s)
    return s


def add(settings, *streets, **kw):
    with session_scope(settings) as s:
        upsert_listings(
            s, [RawListing("csv", st, "Testville", "GA", "30301", **kw) for st in streets]
        )


def run(settings, geocoder, force=False):
    with session_scope(settings) as s:
        lst = list(s.scalars(select(Listing).order_by(Listing.id)))
        report = geocoder.geocode_listings(s, lst, force=force)
        return report, {x.street: (x.geocode_status, x.geocode_source, x.latitude) for x in lst}


def test_census_batch_matches_and_caches(settings):
    add(settings, "1 A St", "2 B St")
    http = FakeHttp(census={"1 A St": (33.1, -84.1), "2 B St": (33.2, -84.2)})
    report, got = run(settings, Geocoder(settings, http))
    assert report.ok == 2 and report.failed == 0
    assert got["1 A St"] == ("ok", "census", 33.1)
    assert len(http.posts) == 1  # one batch call, not one per address
    # second run: nothing left to do -> no network
    report2, _ = run(settings, Geocoder(settings, http))
    assert report2.skipped == 2 and len(http.posts) == 1


def test_cache_is_used_after_a_forced_rerun(settings):
    add(settings, "1 A St")
    http = FakeHttp(census={"1 A St": (33.1, -84.1)})
    run(settings, Geocoder(settings, http))
    http2 = FakeHttp()
    report, _ = run(settings, Geocoder(settings, http2), force=True)
    assert report.from_cache == 1 and not http2.posts


def test_nominatim_fallback_for_census_misses(settings):
    add(settings, "1 A St", "9 New Build Ln")
    http = FakeHttp(census={"1 A St": (33.1, -84.1)}, nominatim={"9 New Build Ln": (33.9, -84.9)})
    slept = []
    report, got = run(settings, Geocoder(settings, http, sleep=slept.append))
    assert got["9 New Build Ln"] == ("ok", "nominatim", 33.9)
    assert len(http.gets) == 1
    assert http.gets[0]["headers"]["User-Agent"].startswith("tests/1.0")
    # no-match results are cached too, so the same miss is not asked again
    add(settings)  # no-op
    http2 = FakeHttp()
    run(settings, Geocoder(settings, http2), force=True)
    assert not http2.posts and not http2.gets


def test_nominatim_is_skipped_without_identifying_user_agent(tmp_path):
    s = Settings(data_dir=tmp_path / "d2", nominatim_user_agent=DEFAULT_NOMINATIM_UA)
    init_db(s)
    add(s, "9 New Build Ln")
    http = FakeHttp(nominatim={"9 New Build Ln": (33.9, -84.9)})
    report, got = run(s, Geocoder(s, http))
    assert got["9 New Build Ln"][0] == "failed" and not http.gets
    assert any("NOMINATIM_USER_AGENT" in m for m in report.messages)


def test_nominatim_cap_and_throttle(settings):
    add(settings, *[f"{i} Miss St" for i in range(1, 6)])
    http = FakeHttp(nominatim={f"{i} Miss St": (33.0 + i / 100, -84.0) for i in range(1, 6)})
    slept = []
    report, got = run(settings, Geocoder(settings, http, sleep=slept.append, nominatim_cap=3))
    assert len(http.gets) == 3
    assert slept and all(t <= 1.1 for t in slept)
    assert any("Stopped the fallback" in m for m in report.messages)


def test_offline_leaves_listings_untouched_and_explains(settings):
    add(settings, "1 A St")
    report, got = run(settings, Geocoder(settings, FakeHttp(fail=True)))
    assert got["1 A St"][0] == "none"
    assert any("Couldn't reach" in m for m in report.messages)


def test_missing_city_state_zip_fails_with_actionable_message(settings):
    with session_scope(settings) as s:
        upsert_listings(s, [RawListing("manual", "1 Lonely Rd")])
    report, got = run(settings, Geocoder(settings, FakeHttp()))
    assert got["1 Lonely Rd"][0] == "failed"
    assert "ZIP" in report.messages[0]


def test_manual_pin_is_never_overwritten_and_is_validated(settings):
    add(settings, "1 A St")
    with session_scope(settings) as s:
        lst = s.scalar(select(Listing))
        Geocoder.set_manual_pin(lst, 33.3, -84.3)
        with pytest.raises(ValueError):
            Geocoder.set_manual_pin(lst, 133.0, 0)
    http = FakeHttp(census={"1 A St": (1.0, 2.0)})
    run(settings, Geocoder(settings, http), force=False)
    with session_scope(settings) as s:
        lst = s.scalar(select(Listing))
        assert (lst.geocode_status, lst.latitude) == ("manual", 33.3)
        assert s.scalar(select(func.count()).select_from(GeocodeCache)) == 0
    assert not http.posts


def test_coordinates_outside_the_us_are_rejected(settings):
    add(settings, "1 A St")
    http = FakeHttp(census={"1 A St": (51.5, 0.1)})  # London: lon is not in the US range
    _, got = run(settings, Geocoder(settings, http, sleep=lambda _: None))
    assert got["1 A St"][0] == "failed"
    json.dumps(got)  # sanity: results are plain data
