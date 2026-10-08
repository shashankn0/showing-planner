"""ManualEntrySource: add listings by typing or pasting addresses."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from core.config import UserPrefs
from sources.addresses import is_plausible_street, normalize_state, parse_full_address
from sources.base import ListingSource, RawListing
from sources.values import classify_access, parse_price, redact_access_codes, to_int


class ManualEntrySource(ListingSource):
    """Build listings from free-text addresses (one per line) or explicit field dicts."""

    name = "manual"

    def __init__(
        self,
        addresses: Iterable[str] = (),
        records: Iterable[dict] = (),
        prefs: UserPrefs | None = None,
    ) -> None:
        self.addresses = list(addresses)
        self.records = list(records)
        self.prefs = prefs or UserPrefs()
        self.skipped: list[str] = []

    @classmethod
    def from_text(cls, text: str, prefs: UserPrefs | None = None) -> ManualEntrySource:
        return cls(addresses=[ln for ln in text.splitlines() if ln.strip()], prefs=prefs)

    def fetch(self) -> Iterator[RawListing]:
        self.skipped = []
        for line in self.addresses:
            parsed = parse_full_address(line)
            if not is_plausible_street(
                parsed.street, parsed.city, parsed.state, parsed.postal_code
            ):
                self.skipped.append(line)
                continue
            yield RawListing(
                source=self.name,
                street=parsed.street or "",
                unit=parsed.unit,
                city=parsed.city or self.prefs.default_city or None,
                state=parsed.state or normalize_state(self.prefs.default_state),
                postal_code=parsed.postal_code,
            )
        for rec in self.records:
            addr = rec.get("address") or rec.get("street") or ""
            parsed = parse_full_address(addr)
            if not is_plausible_street(
                parsed.street,
                parsed.city or rec.get("city"),
                parsed.state or rec.get("state"),
                parsed.postal_code or rec.get("postal_code"),
            ):
                self.skipped.append(str(addr))
                continue
            showing = redact_access_codes(rec.get("showing_requirements"))
            yield RawListing(
                source=self.name,
                street=parsed.street or "",
                unit=rec.get("unit") or parsed.unit,
                city=rec.get("city") or parsed.city or self.prefs.default_city or None,
                state=rec.get("state") or parsed.state or normalize_state(self.prefs.default_state),
                postal_code=rec.get("postal_code") or parsed.postal_code,
                source_listing_id=(
                    str(rec["source_listing_id"]).strip().upper()
                    if rec.get("source_listing_id")
                    else None
                ),
                list_price=parse_price(rec.get("list_price")),
                beds=to_int(rec.get("beds")),
                sqft=to_int(rec.get("sqft")),
                showing_requirements=showing,
                access_type=rec.get("access_type") or classify_access(showing),
            )
