"""The ListingSource contract. Every way of getting listings in goes through this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from sources.addresses import address_key, format_full


@dataclass
class RawListing:
    """A listing as produced by any source, already normalized. Only the address is required."""

    source: str
    street: str
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    unit: str | None = None
    source_listing_id: str | None = None
    county: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    list_price: int | None = None
    status: str | None = None
    property_type: str | None = None
    beds: int | None = None
    baths_full: int | None = None
    baths_half: int | None = None
    baths_total: float | None = None
    sqft: int | None = None
    lot_sqft: int | None = None
    year_built: int | None = None
    list_date: date | None = None
    showing_requirements: str | None = None
    access_type: str = "unknown"
    is_synthetic: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def address_full(self) -> str:
        return format_full(self.street, self.unit, self.city, self.state, self.postal_code)

    @property
    def address_key(self) -> str:
        return address_key(self.street, self.unit, self.city, self.state, self.postal_code)

    @property
    def dedupe_key(self) -> str:
        if self.source_listing_id:
            return f"mls:{self.source_listing_id.strip().upper()}"
        return self.address_key


class ListingSource(ABC):
    """Anything that can produce listings: a CSV export, manual entry, a licensed feed."""

    name: str = "source"

    @abstractmethod
    def fetch(self) -> Iterable[RawListing]:
        """Yield normalized listings. Must not scrape sites that prohibit it."""
