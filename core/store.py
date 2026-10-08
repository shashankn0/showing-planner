"""Persist listings from any ListingSource, de-duplicating across imports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import ImportBatch, Listing
from sources.base import RawListing

# Fields copied from a RawListing onto a Listing when the raw value is present.
_COPY_FIELDS = (
    "street", "unit", "city", "state", "postal_code", "county", "list_price", "status",
    "property_type", "beds", "baths_full", "baths_half", "baths_total", "sqft", "lot_sqft",
    "year_built", "list_date", "showing_requirements",
)  # fmt: skip


@dataclass
class UpsertResult:
    new: int = 0
    updated: int = 0
    listing_ids: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.new + self.updated


def _find_existing(session: Session, raw: RawListing) -> Listing | None:
    hit = session.scalar(select(Listing).where(Listing.dedupe_key == raw.dedupe_key))
    if hit:
        return hit
    if raw.source_listing_id:
        # An earlier address-only listing (typed in, or from a bare address list) gains its MLS#.
        return session.scalar(
            select(Listing).where(
                Listing.address_key == raw.address_key, Listing.source_listing_id.is_(None)
            )
        )
    # No MLS#: attach to an existing listing at the same address rather than duplicating it.
    return session.scalar(select(Listing).where(Listing.address_key == raw.address_key))


def upsert_listings(
    session: Session, raws: Iterable[RawListing], batch: ImportBatch | None = None
) -> UpsertResult:
    result = UpsertResult()
    seen: dict[str, Listing] = {}  # listings created/updated earlier in this same call
    for raw in raws:
        existing = seen.get(raw.dedupe_key) or _find_existing(session, raw)
        if existing is None:
            lst = Listing(
                source=raw.source,
                dedupe_key=raw.dedupe_key,
                address_key=raw.address_key,
                source_listing_id=raw.source_listing_id,
                address_full=raw.address_full,
                access_type=raw.access_type,
                is_synthetic=raw.is_synthetic,
                import_batch_id=batch.id if batch else None,
            )
            for f in _COPY_FIELDS:
                setattr(lst, f, getattr(raw, f))
            _set_coords(lst, raw)
            session.add(lst)
            session.flush()
            result.new += 1
        else:
            lst = existing
            address_changed = lst.address_key != raw.address_key
            for f in _COPY_FIELDS:
                value = getattr(raw, f)
                if value is not None:
                    setattr(lst, f, value)
            if raw.source_listing_id and not lst.source_listing_id:
                lst.source_listing_id = raw.source_listing_id
                lst.dedupe_key = raw.dedupe_key
            if raw.access_type != "unknown":
                lst.access_type = raw.access_type
            lst.address_key, lst.address_full = raw.address_key, raw.address_full
            if address_changed and lst.geocode_status != "manual":
                lst.latitude = lst.longitude = None
                lst.geocode_status, lst.geocode_source = "none", None
            _set_coords(lst, raw)
            if batch:
                lst.import_batch_id = batch.id
            result.updated += 1
        seen[raw.dedupe_key] = lst
        result.listing_ids.append(lst.id)
    if batch:
        batch.rows_new, batch.rows_updated = result.new, result.updated
    return result


def _set_coords(lst: Listing, raw: RawListing) -> None:
    """Imported coordinates never override a pin the user placed by hand."""
    if raw.latitude is None or raw.longitude is None or lst.geocode_status == "manual":
        return
    lst.latitude, lst.longitude = raw.latitude, raw.longitude
    lst.geocode_status, lst.geocode_source = "import", "file"
