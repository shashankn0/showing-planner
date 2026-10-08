"""UI-independent actions used by the Streamlit pages (so they can be tested without a browser)."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from core.config import Settings
from core.db import session_scope
from core.geocoding import Geocoder, GeocodeReport, Http
from core.models import ImportBatch, Listing, Photo
from core.store import upsert_listings
from sources.base import RawListing
from sources.csv_import import CsvImportSource, ImportAnalysis
from sources.manual import ManualEntrySource
from sources.photos import (
    IngestResult,
    MatchReport,
    PhotoGroup,
    extract_zip,
    ingest_photos,
    listings_from_groups,
    match_groups,
    scan_photo_folder,
)
from sources.profiles import save_profile

NEW_LISTING = "__new__"


@dataclass
class ImportSummary:
    new: int = 0
    updated: int = 0
    skipped: list[tuple[int, str]] = field(default_factory=list)
    assumed_defaults: int = 0
    geocode: GeocodeReport | None = None
    listing_ids: list[int] = field(default_factory=list)


def preview_rows(analysis: ImportAnalysis, settings: Settings, n: int = 8) -> pd.DataFrame:
    """First rows as they will be imported, for the user to sanity-check the mapping."""
    src = CsvImportSource.from_analysis(analysis, settings.prefs)
    rows = []
    for raw in src.fetch():
        rows.append(_raw_to_preview(raw))
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def _raw_to_preview(raw: RawListing) -> dict:
    baths = raw.baths_total
    return {
        "MLS #": raw.source_listing_id or "",
        "Address": raw.address_full,
        "Price": raw.list_price,
        "Beds": raw.beds,
        "Baths": baths,
        "Sq ft": raw.sqft,
        "Status": raw.status or "",
        "Access": raw.access_type,
        "Has coordinates": raw.latitude is not None,
    }


def run_import(
    settings: Settings,
    analysis: ImportAnalysis,
    filename: str,
    *,
    is_synthetic: bool = False,
    save_layout_as: str | None = None,
    geocode: bool = True,
    http: Http | None = None,
) -> ImportSummary:
    if not analysis.is_importable:
        raise ValueError("No address column is selected, so there is nothing to import.")
    summary = ImportSummary()
    src = CsvImportSource.from_analysis(analysis, settings.prefs, is_synthetic)
    with session_scope(settings) as s:
        batch = ImportBatch(
            source="csv",
            filename=filename,
            profile_name=save_layout_as or analysis.profile_name,
            mapping_json=analysis.mapping,
            rows_total=len(analysis.table.rows),
        )
        s.add(batch)
        s.flush()
        res = upsert_listings(s, src.fetch(), batch)
        batch.rows_skipped = len(src.skipped)
    summary.new, summary.updated = res.new, res.updated
    summary.skipped, summary.assumed_defaults = src.skipped, src.assumed_defaults
    summary.listing_ids = res.listing_ids
    if save_layout_as:
        save_profile(
            settings.mappings_dir, save_layout_as, analysis.mapping, analysis.table.headers
        )
    if geocode and res.listing_ids:
        summary.geocode = geocode_listing_ids(settings, res.listing_ids, http=http)
    return summary


def add_addresses(
    settings: Settings, text: str, *, geocode: bool = True, http: Http | None = None
) -> ImportSummary:
    src = ManualEntrySource.from_text(text, settings.prefs)
    summary = ImportSummary()
    with session_scope(settings) as s:
        batch = ImportBatch(source="manual", rows_total=len(src.addresses))
        s.add(batch)
        s.flush()
        res = upsert_listings(s, src.fetch(), batch)
        batch.rows_skipped = len(src.skipped)
    summary.new, summary.updated, summary.listing_ids = res.new, res.updated, res.listing_ids
    summary.skipped = [
        (i, f"couldn't read an address from: {t}") for i, t in enumerate(src.skipped, 1)
    ]
    if geocode and res.listing_ids:
        summary.geocode = geocode_listing_ids(settings, res.listing_ids, http=http)
    return summary


def geocode_listing_ids(
    settings: Settings,
    ids: list[int] | None = None,
    *,
    force: bool = False,
    http: Http | None = None,
) -> GeocodeReport:
    with session_scope(settings) as s:
        q = select(Listing)
        if ids is not None:
            q = q.where(Listing.id.in_(ids))
        return Geocoder(settings, http).geocode_listings(s, list(s.scalars(q)), force=force)


# ------------------------------------------------------------------ photos


@dataclass
class PhotoScan:
    root: Path
    groups: list[PhotoGroup]
    report: MatchReport
    address_by_id: dict[int, str]
    new_listing_keys: set[str]  # folder names that look like addresses (can create a listing)
    cleanup_dir: Path | None = None


def scan_photos(
    settings: Settings, folder: Path | None = None, zip_bytes: bytes | None = None
) -> PhotoScan:
    cleanup = None
    if zip_bytes is not None:
        cleanup = settings.data_dir / "_incoming" / str(int(time.time()))
        root = extract_zip(zip_bytes, cleanup)
    elif folder is not None:
        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValueError(f"'{root}' is not a folder on this computer.")
    else:
        raise ValueError("Choose a folder or upload a zip.")
    groups = scan_photo_folder(root)
    with session_scope(settings) as s:
        listings = list(s.scalars(select(Listing)))
        report = match_groups(groups, listings)
        addresses = {lst.id: lst.address_full for lst in listings}
    can_create = {g.key for g in report.unmatched if listings_from_groups([g])[0]}
    return PhotoScan(root, groups, report, addresses, can_create, cleanup)


def apply_photos(
    settings: Settings,
    scan: PhotoScan,
    assignments: dict[str, int | str | None],
    mode: str = "copy",
) -> IngestResult:
    """`assignments` maps group key -> listing id, NEW_LISTING, or None (skip)."""
    total = IngestResult()
    by_key = {g.key: g for g in scan.groups}
    with session_scope(settings) as s:
        for key, target in assignments.items():
            group = by_key.get(key)
            if group is None or target is None:
                continue
            if target == NEW_LISTING:
                made, _ = listings_from_groups([group])
                if not made:
                    continue
                res = upsert_listings(s, made)
                target = res.listing_ids[0]
            listing = s.get(Listing, target)
            if listing is None:
                continue
            r = ingest_photos(s, settings, listing, group.files, mode=mode)
            total.added += r.added
            total.duplicates += r.duplicates
            total.unreadable += r.unreadable
    if scan.cleanup_dir and scan.cleanup_dir.exists() and mode == "copy":
        shutil.rmtree(scan.cleanup_dir, ignore_errors=True)
    return total


# ------------------------------------------------------------------ tables


def listings_table(settings: Settings) -> pd.DataFrame:
    with session_scope(settings) as s:
        counts = dict(
            s.execute(select(Photo.listing_id, func.count()).group_by(Photo.listing_id)).all()
        )
        rows = [
            {
                "ID": lst.id,
                "MLS #": lst.source_listing_id or "",
                "Address": lst.address_full,
                "Price": lst.list_price,
                "Beds": lst.beds,
                "Baths": lst.baths_total,
                "Sq ft": lst.sqft,
                "Status": lst.status or "",
                "Access": lst.access_type,
                "Photos": counts.get(lst.id, 0),
                "Map": lst.geocode_status,
                "Demo data": lst.is_synthetic,
            }
            for lst in s.scalars(select(Listing).order_by(Listing.id.desc()))
        ]
    return pd.DataFrame(rows)
