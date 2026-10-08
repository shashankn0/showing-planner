"""RentCast sale-listings source (optional, licensed API; bring your own key).

RentCast's terms say an API key may not be shared with other people for their own use, so a
public/hosted deployment must never use the site owner's key: each user supplies their own, and it
is held only for their session. Storing and displaying the data is permitted by their API terms.
The documented listing fields include no photos, so listings from here have none.

Docs: https://developers.rentcast.io/reference/sale-listings
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import Any, Protocol

import requests

from sources.addresses import UNIT_WORDS, clean, normalize_state, normalize_zip
from sources.base import ListingSource, RawListing
from sources.values import is_missing, parse_baths, parse_date, to_float, to_int

BASE_URL = "https://api.rentcast.io/v1"
SALE_LISTINGS_PATH = "/listings/sale"
MAX_LIMIT = 500
SIGNUP_URL = "https://www.rentcast.io/api"


class RentCastError(Exception):
    """A problem with a RentCast request, worded for the person using the app."""


class Http(Protocol):
    def get(self, url: str, **kw): ...  # noqa: ANN003, ANN201


def _status(value: object) -> str | None:
    s = clean(value).lower()
    if not s:
        return None
    return "active" if s == "active" else "off_market" if s == "inactive" else s[:32]


def listing_from_payload(item: dict[str, Any]) -> RawListing | None:
    """Map one RentCast listing object to a RawListing (None if it has no usable address)."""
    street = clean(item.get("addressLine1"))
    if not street:
        formatted = clean(item.get("formattedAddress"))
        street = formatted.split(",")[0].strip() if formatted else ""
    if len(street) < 3:
        return None
    unit = clean(item.get("addressLine2"))
    unit = clean(re.sub(rf"^{UNIT_WORDS}\.?\s*#?\s*", "", unit, flags=re.I)).upper() or None
    lat, lon = to_float(item.get("latitude")), to_float(item.get("longitude"))
    if lat is None or lon is None or not (15 <= lat <= 72 and -180 <= lon <= -60):
        lat = lon = None
    full, half, total = parse_baths(item.get("bathrooms"))
    mls = clean(item.get("mlsNumber")).upper()
    return RawListing(
        source="rentcast",
        street=street,
        unit=unit,
        city=clean(item.get("city")) or None,
        state=normalize_state(item.get("state")),
        postal_code=normalize_zip(item.get("zipCode")),
        source_listing_id=mls or None,
        county=clean(item.get("county")) or None,
        latitude=lat,
        longitude=lon,
        list_price=to_int(item.get("price")),
        status=_status(item.get("status")),
        property_type=clean(item.get("propertyType")) or None,
        beds=to_int(item.get("bedrooms")),
        baths_full=full,
        baths_half=half,
        baths_total=total,
        sqft=to_int(item.get("squareFootage")),
        lot_sqft=to_int(item.get("lotSize")),
        year_built=to_int(item.get("yearBuilt")),
        list_date=parse_date(item.get("listedDate")),
    )


class RentCastSource(ListingSource):
    name = "rentcast"

    def __init__(
        self,
        api_key: str,
        *,
        city: str | None = None,
        state: str | None = None,
        zip_code: str | None = None,
        max_price: int | None = None,
        property_types: list[str] | tuple[str, ...] = (),
        status: str = "Active",
        limit: int = 200,
        http: Http | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.city = clean(city)
        self.state = normalize_state(state) or ""
        self.zip_code = normalize_zip(zip_code) or ""
        self.max_price = max_price
        self.property_types = list(property_types)
        self.status = status
        self.limit = max(1, min(int(limit), MAX_LIMIT))
        self.http = http or requests.Session()
        self.skipped = 0

    # ------------------------------------------------------------------ request

    def request_params(self) -> dict[str, Any]:
        """Query parameters (never includes the API key, so it is safe to cache/log)."""
        if not (self.zip_code or (self.city and self.state)):
            raise RentCastError("Enter a ZIP code, or a city and its two-letter state.")
        params: dict[str, Any] = {"status": self.status, "limit": self.limit}
        if self.zip_code:
            params["zipCode"] = self.zip_code
        else:
            params["city"], params["state"] = self.city, self.state
        if self.max_price:
            params["price"] = f"0-{int(self.max_price)}"
        if len(self.property_types) == 1:  # more than one: filter afterwards instead
            params["propertyType"] = self.property_types[0]
        return params

    def fetch_payload(self) -> list[dict[str, Any]]:
        """One API call. Raises RentCastError with a plain-language message on any problem."""
        if not self.api_key:
            raise RentCastError(f"Paste your RentCast API key first (get one at {SIGNUP_URL}).")
        params = self.request_params()
        try:
            resp = self.http.get(
                BASE_URL + SALE_LISTINGS_PATH,
                params=params,
                headers={"X-Api-Key": self.api_key, "Accept": "application/json"},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RentCastError(
                "Couldn't reach RentCast (offline, or their service is busy). "
                f"Nothing was used from your quota. ({type(exc).__name__})"
            ) from None
        code = resp.status_code
        if code in (401, 403):
            raise RentCastError("RentCast didn't accept that API key. Check it and try again.")
        if code == 429:
            raise RentCastError(
                "RentCast says you've hit a request limit (likely the monthly free allowance)."
            )
        if code >= 400:
            detail = ""
            with contextlib.suppress(ValueError, AttributeError):
                detail = clean(resp.json().get("message"))
            raise RentCastError(f"RentCast returned an error ({code}). {detail}".strip())
        try:
            data = resp.json()
        except ValueError:
            raise RentCastError("RentCast sent back something unreadable.") from None
        if not isinstance(data, list):
            raise RentCastError("RentCast sent back an unexpected response.")
        return data

    # ------------------------------------------------------------------ mapping

    def fetch(self) -> Iterator[RawListing]:
        yield from self.listings_from_payload(self.fetch_payload())

    def listings_from_payload(self, payload: list[dict[str, Any]]) -> Iterator[RawListing]:
        self.skipped = 0
        wanted = {t.lower() for t in self.property_types}
        for item in payload:
            if not isinstance(item, dict):
                self.skipped += 1
                continue
            raw = listing_from_payload(item)
            if raw is None or is_missing(raw.street):
                self.skipped += 1
                continue
            if len(wanted) > 1 and (raw.property_type or "").lower() not in wanted:
                continue
            yield raw
