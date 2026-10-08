"""CsvImportSource: import listings from any CSV/TSV/Excel export, no fixed layout required."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from core.config import UserPrefs
from sources.addresses import (
    ParsedAddress,
    clean,
    compose_street,
    is_plausible_street,
    normalize_state,
    normalize_zip,
    parse_full_address,
    split_unit,
)
from sources.base import ListingSource, RawListing
from sources.columns import FIELDS, MappingResult, detect_mapping
from sources.profiles import find_profile
from sources.tabular import Table, read_table
from sources.values import (
    classify_access,
    is_missing,
    normalize_status,
    parse_baths,
    parse_date,
    parse_price,
    redact_access_codes,
    to_float,
    to_int,
)

ACRES_TO_SQFT = 43_560


@dataclass
class ImportAnalysis:
    """Everything the UI needs to show a mapping for confirmation before importing."""

    table: Table
    detection: MappingResult
    mapping: dict[str, str]  # effective: saved profile > auto-detected, then user overrides
    profile_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_importable(self) -> bool:
        m = self.mapping
        return "address" in m or ("street_name" in m and "street_number" in m)


def analyze_table(
    table: Table,
    profiles_dir: Path | None = None,
    overrides: dict[str, str | None] | None = None,
) -> ImportAnalysis:
    detection = detect_mapping(table.headers, table.rows)
    mapping = dict(detection.mapping)
    profile_name = None
    if profiles_dir is not None and not table.synthetic_headers:
        prof = find_profile(profiles_dir, table.headers)
        if prof:
            mapping, profile_name = dict(prof.mapping), prof.name
    for fname, col in (overrides or {}).items():
        if col is None or col == "":
            mapping.pop(fname, None)
        elif col in table.headers:
            # one column can only feed one field
            for other in [k for k, v in mapping.items() if v == col and k != fname]:
                mapping.pop(other)
            mapping[fname] = col
    warnings = list(table.notes) + list(detection.warnings)
    return ImportAnalysis(table, detection, mapping, profile_name, warnings)


def analyze_file(
    path_or_bytes: Path | str | bytes,
    filename: str | None = None,
    profiles_dir: Path | None = None,
    overrides: dict[str, str | None] | None = None,
) -> ImportAnalysis:
    return analyze_table(read_table(path_or_bytes, filename), profiles_dir, overrides)


class CsvImportSource(ListingSource):
    name = "csv"

    def __init__(
        self,
        table: Table,
        mapping: dict[str, str],
        prefs: UserPrefs | None = None,
        is_synthetic: bool = False,
    ) -> None:
        self.table = table
        self.mapping = {k: v for k, v in mapping.items() if k in FIELDS}
        self.prefs = prefs or UserPrefs()
        self.is_synthetic = is_synthetic
        self.skipped: list[tuple[int, str]] = []  # (1-based data row number, reason)
        self.assumed_defaults = 0
        self._idx = {h: i for i, h in enumerate(table.headers)}

    @classmethod
    def from_analysis(
        cls, analysis: ImportAnalysis, prefs: UserPrefs | None = None, is_synthetic: bool = False
    ) -> CsvImportSource:
        return cls(analysis.table, analysis.mapping, prefs, is_synthetic)

    # ------------------------------------------------------------------ rows

    def fetch(self) -> Iterator[RawListing]:
        self.skipped, self.assumed_defaults = [], 0
        for n, row in enumerate(self.table.rows, start=1):
            listing = self._row_to_listing(row, n)
            if listing is not None:
                yield listing

    def _cell(self, row: list[str], fname: str) -> str:
        col = self.mapping.get(fname)
        if col is None:
            return ""
        i = self._idx.get(col)
        if i is None or i >= len(row):
            return ""
        v = row[i].strip()
        return "" if is_missing(v) else v

    def _row_to_listing(self, row: list[str], n: int) -> RawListing | None:
        g = lambda f: self._cell(row, f)  # noqa: E731

        parsed = parse_full_address(g("address")) if g("address") else ParsedAddress()
        street = parsed.street
        unit = parsed.unit
        if "street_name" in self.mapping and "street_number" in self.mapping:
            composed = compose_street(
                g("street_number"),
                g("street_dir_prefix"),
                g("street_name"),
                g("street_suffix"),
                g("street_dir_suffix"),
            )
            if g("street_number") and g("street_name"):
                street, unit = composed, unit
        if unit is None and street:
            street, unit = split_unit(street)
        unit = clean(g("unit")).upper() or unit

        city = clean(g("city")) or parsed.city
        state = normalize_state(g("state")) or parsed.state
        zip_ = normalize_zip(g("postal_code")) or parsed.postal_code
        if not is_plausible_street(street, city, state, zip_):
            self.skipped.append((n, "no street address"))
            return None
        assumed = False
        if not state and self.prefs.default_state:
            state, assumed = normalize_state(self.prefs.default_state), True
        if not city and self.prefs.default_city:
            city, assumed = clean(self.prefs.default_city), True
        if assumed:
            self.assumed_defaults += 1

        full, half, total = self._baths(g)
        lat, lon = to_float(g("latitude")), to_float(g("longitude"))
        if lat is not None and lon is not None and not (15 <= lat <= 72 and -180 <= lon <= -60):
            lat = lon = None  # outside the US: ignore rather than trust a bad column
        elif (lat is None) != (lon is None):
            lat = lon = None

        lot = to_int(g("lot_sqft"))
        if lot is None and g("lot_acres"):
            acres = to_float(g("lot_acres"))
            lot = int(acres * ACRES_TO_SQFT) if acres is not None else None

        showing = redact_access_codes(g("showing_requirements"))
        mls = g("source_listing_id")
        return RawListing(
            source=self.name,
            street=street,
            unit=unit or None,
            city=city or None,
            state=state,
            postal_code=zip_,
            source_listing_id=mls.strip().upper() if mls else None,
            county=clean(g("county")) or None,
            latitude=lat,
            longitude=lon,
            list_price=parse_price(g("list_price")),
            status=normalize_status(g("status")),
            property_type=clean(g("property_type")) or None,
            beds=to_int(g("beds")),
            baths_full=full,
            baths_half=half,
            baths_total=total,
            sqft=to_int(g("sqft")),
            lot_sqft=lot,
            year_built=self._year(g("year_built")),
            list_date=parse_date(g("list_date")),
            showing_requirements=showing,
            access_type=classify_access(showing),
            is_synthetic=self.is_synthetic,
        )

    @staticmethod
    def _year(value: str) -> int | None:
        y = to_int(value)
        return y if y is not None and 1700 <= y <= 2100 else None

    def _baths(self, g) -> tuple[int | None, int | None, float | None]:  # noqa: ANN001
        if self.mapping.get("baths_full") or self.mapping.get("baths_half"):
            full, half = to_int(g("baths_full")), to_int(g("baths_half"))
            if full is not None or half is not None:
                full, half = full or 0, half or 0
                return full, half, full + 0.5 * half
        return parse_baths(g("baths_total"))
