"""Geocoding with a persistent cache.

Order of preference: coordinates already in the import file -> Census Geocoder (free, no key,
batch) -> Nominatim (fallback, throttled, only with an identifying User-Agent) -> manual pin.
Every provider answer, including "no match", is cached so addresses are never asked twice.
"""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import DEFAULT_NOMINATIM_UA, Settings
from core.models import GeocodeCache, Listing
from sources.addresses import clean, is_plausible_street, normalize_street, parse_full_address

CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
CENSUS_BENCHMARK = "Public_AR_Current"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CENSUS_CHUNK = 1000  # the service allows 10,000; smaller chunks fail more gracefully
NOMINATIM_MIN_INTERVAL = 1.1  # policy: at most 1 request per second
US_LAT, US_LON = (15.0, 72.0), (-180.0, -60.0)


class Http(Protocol):
    def get(self, url: str, **kw): ...  # noqa: ANN003, ANN201
    def post(self, url: str, **kw): ...  # noqa: ANN003, ANN201


@dataclass
class GeocodeReport:
    ok: int = 0
    failed: int = 0
    from_cache: int = 0
    skipped: int = 0
    messages: list[str] = field(default_factory=list)


def query_key(street: str | None, city: str | None, state: str | None, zip_: str | None) -> str:
    return "|".join(
        [normalize_street(street or ""), clean(city).upper(), (state or "").upper(), zip_ or ""]
    )


def _in_us(lat: float, lon: float) -> bool:
    return US_LAT[0] <= lat <= US_LAT[1] and US_LON[0] <= lon <= US_LON[1]


class Geocoder:
    def __init__(
        self,
        settings: Settings,
        http: Http | None = None,
        sleep: Callable[[float], None] = time.sleep,
        nominatim_cap: int = 25,
    ) -> None:
        self.settings = settings
        self.http = http or requests.Session()
        self.sleep = sleep
        self.nominatim_cap = nominatim_cap
        self._last_nominatim = 0.0

    @property
    def nominatim_enabled(self) -> bool:
        ua = self.settings.nominatim_user_agent
        return bool(ua) and ua != DEFAULT_NOMINATIM_UA

    # ------------------------------------------------------------------ public

    def geocode_listings(
        self, session: Session, listings: Iterable[Listing], force: bool = False
    ) -> GeocodeReport:
        report = GeocodeReport()
        todo: list[Listing] = []
        for lst in listings:
            if lst.geocode_status in {"manual", "import", "ok"} and not force:
                report.skipped += 1
            elif not (lst.street and (lst.postal_code or (lst.city and lst.state))):
                lst.geocode_status = "failed"
                report.failed += 1
                report.messages.append(
                    f"{lst.address_full}: needs a ZIP code, or a city and state, to look up."
                )
            else:
                todo.append(lst)

        # 1. cache, then Census batch for the rest
        pending: list[Listing] = []
        for lst in todo:
            hit = self._cached(session, "census", lst)
            if hit is None:
                pending.append(lst)
            else:
                self._apply(lst, hit, "census", report, cached=True)
        misses = self._census_batch(session, pending, report) if pending else []

        # 2. Nominatim fallback for Census misses
        if misses:
            self._nominatim_fallback(session, misses, report)
        return report

    def geocode_text(self, session: Session, text: str) -> tuple[float, float] | None:
        """Look up one free-text address (e.g. a client's workplace). Cached like any other."""
        parsed = parse_full_address(
            text, self.settings.prefs.default_city, self.settings.prefs.default_state
        )
        if not is_plausible_street(parsed.street, parsed.city, parsed.state, parsed.postal_code):
            return None
        tmp = Listing(
            street=parsed.street,
            city=parsed.city,
            state=parsed.state,
            postal_code=parsed.postal_code,
            address_full=clean(text),
            geocode_status="none",
        )  # transient: never added to the session
        self.geocode_listings(session, [tmp], force=True)
        if tmp.latitude is None or tmp.longitude is None:
            return None
        return tmp.latitude, tmp.longitude

    @staticmethod
    def set_manual_pin(listing: Listing, lat: float, lon: float) -> None:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Latitude must be -90..90 and longitude -180..180.")
        listing.latitude, listing.longitude = lat, lon
        listing.geocode_status, listing.geocode_source = "manual", "user"

    # ------------------------------------------------------------------ cache

    @staticmethod
    def _key(lst: Listing) -> str:
        return query_key(lst.street, lst.city, lst.state, lst.postal_code)

    def _cached(self, session: Session, provider: str, lst: Listing) -> GeocodeCache | None:
        return session.scalar(
            select(GeocodeCache).where(
                GeocodeCache.provider == provider, GeocodeCache.query_key == self._key(lst)
            )
        )

    def _store(
        self,
        session: Session,
        provider: str,
        lst: Listing,
        status: str,
        lat: float | None = None,
        lon: float | None = None,
        matched: str | None = None,
    ) -> GeocodeCache:
        row = self._cached(session, provider, lst) or GeocodeCache(
            provider=provider, query_key=self._key(lst), status=status
        )
        row.status, row.latitude, row.longitude, row.matched_address = status, lat, lon, matched
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def _apply(
        lst: Listing, row: GeocodeCache, provider: str, report: GeocodeReport, cached: bool
    ) -> bool:
        if row.status == "ok" and row.latitude is not None and row.longitude is not None:
            lst.latitude, lst.longitude = row.latitude, row.longitude
            lst.geocode_status, lst.geocode_source = "ok", provider
            report.ok += 1
            report.from_cache += int(cached)
            return True
        return False

    # ------------------------------------------------------------------ census

    def _census_batch(
        self, session: Session, listings: list[Listing], report: GeocodeReport
    ) -> list[Listing]:
        """Return the listings Census could not match (or that need a retry)."""
        unmatched: list[Listing] = []
        for start in range(0, len(listings), CENSUS_CHUNK):
            chunk = listings[start : start + CENSUS_CHUNK]
            buf = io.StringIO()
            w = csv.writer(buf)
            for i, lst in enumerate(chunk):
                w.writerow([i, lst.street, lst.city or "", lst.state or "", lst.postal_code or ""])
            try:
                resp = self.http.post(
                    CENSUS_BATCH_URL,
                    data={"benchmark": CENSUS_BENCHMARK},
                    files={"addressFile": ("addresses.csv", buf.getvalue())},
                    timeout=120,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                report.messages.append(
                    "Couldn't reach the Census geocoder (offline, or the service is busy). "
                    f"Nothing was changed; try again later. ({type(exc).__name__})"
                )
                return unmatched  # leave the rest untouched so a retry picks them up
            answered: set[int] = set()
            for row in csv.reader(io.StringIO(resp.text)):
                if len(row) < 3 or not row[0].strip().isdigit():
                    continue
                idx = int(row[0])
                if idx >= len(chunk):
                    continue
                answered.add(idx)
                lst = chunk[idx]
                coords = _parse_census_coords(row)
                if row[2] == "Match" and coords:
                    cached = self._store(
                        session,
                        "census",
                        lst,
                        "ok",
                        coords[0],
                        coords[1],
                        row[4] if len(row) > 4 else None,
                    )
                    self._apply(lst, cached, "census", report, cached=False)
                else:
                    self._store(session, "census", lst, "no_match")
                    unmatched.append(lst)
            for idx, lst in enumerate(chunk):
                if idx not in answered:
                    unmatched.append(lst)  # dropped from the response: retry via fallback
        return unmatched

    # ------------------------------------------------------------------ nominatim

    def _nominatim_fallback(
        self, session: Session, listings: list[Listing], report: GeocodeReport
    ) -> None:
        asked = 0
        for lst in listings:
            hit = self._cached(session, "nominatim", lst)
            if hit is not None:
                if not self._apply(lst, hit, "nominatim", report, cached=True):
                    self._mark_failed(lst, report)
                continue
            if not self.nominatim_enabled:
                self._mark_failed(lst, report)
                continue
            if asked >= self.nominatim_cap:
                report.messages.append(
                    f"Stopped the fallback lookup after {self.nominatim_cap} addresses to respect "
                    "the free service's limits. Run it again later, or place pins manually."
                )
                break
            asked += 1
            self._throttle()
            try:
                resp = self.http.get(
                    NOMINATIM_URL,
                    params={
                        "street": lst.street,
                        "city": lst.city or "",
                        "state": lst.state or "",
                        "postalcode": lst.postal_code or "",
                        "countrycodes": "us",
                        "format": "json",
                        "limit": 1,
                    },
                    headers={"User-Agent": self.settings.nominatim_user_agent},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                report.messages.append(f"Fallback lookup failed for {lst.address_full}: {exc}")
                continue
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                if _in_us(lat, lon):
                    row = self._store(
                        session, "nominatim", lst, "ok", lat, lon, data[0].get("display_name")
                    )
                    self._apply(lst, row, "nominatim", report, cached=False)
                    continue
            self._store(session, "nominatim", lst, "no_match")
            self._mark_failed(lst, report)
        if not self.nominatim_enabled and listings:
            report.messages.append(
                "Some addresses weren't found by the Census geocoder. To try a second free "
                "service, set NOMINATIM_USER_AGENT in .env to an identifying string with your "
                "contact info, or place pins manually."
            )

    def _throttle(self) -> None:
        wait = NOMINATIM_MIN_INTERVAL - (time.monotonic() - self._last_nominatim)
        if wait > 0:
            self.sleep(wait)
        self._last_nominatim = time.monotonic()

    @staticmethod
    def _mark_failed(lst: Listing, report: GeocodeReport) -> None:
        lst.geocode_status = "failed"
        report.failed += 1
        report.messages.append(f"{lst.address_full}: not found. Place a pin manually.")


def _parse_census_coords(row: list[str]) -> tuple[float, float] | None:
    if len(row) < 6 or "," not in row[5]:
        return None
    try:
        lon_s, lat_s = row[5].split(",", 1)
        lon, lat = float(lon_s), float(lat_s)
    except ValueError:
        return None
    return (lat, lon) if _in_us(lat, lon) else None
