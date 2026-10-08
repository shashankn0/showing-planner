"""UI-independent actions used by the Streamlit pages (so they can be tested without a browser)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from core.config import Settings
from core.db import session_scope
from core.geocoding import Geocoder, GeocodeReport, Http
from core.models import Client, ImportBatch, Listing, ListingScore, Photo
from core.store import upsert_listings
from ranking.engine import Candidate, RankingResult, rank
from ranking.preferences import Destination, Preferences
from sources.base import RawListing
from sources.csv_import import CsvImportSource, ImportAnalysis
from sources.demo import demo_listings
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
from sources.rentcast import Http as RentCastHttp
from sources.rentcast import RentCastSource

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


# ------------------------------------------------------------------ ranking ("Find Homes")


def candidates_from_db(settings: Settings) -> list[Candidate]:
    """Listings reduced to the facts the ranking may use, plus photo condition when scored."""
    with session_scope(settings) as s:
        latest: dict[int, ListingScore] = {}
        for sc in s.scalars(select(ListingScore).order_by(ListingScore.created_at)):
            latest[sc.listing_id] = sc
        out = []
        for lst in s.scalars(select(Listing).order_by(Listing.id)):
            sc = latest.get(lst.id)
            out.append(
                Candidate(
                    id=lst.id,
                    address=lst.address_full,
                    price=lst.list_price,
                    beds=lst.beds,
                    baths=lst.baths_total,
                    sqft=lst.sqft,
                    lot_sqft=lst.lot_sqft,
                    year_built=lst.year_built,
                    property_type=lst.property_type,
                    status=lst.status,
                    lat=lst.latitude,
                    lon=lst.longitude,
                    list_date=lst.list_date,
                    condition=sc.overall if sc else None,
                    condition_photos=sc.n_photos_used if sc else 0,
                    mls=lst.source_listing_id,
                    is_synthetic=lst.is_synthetic,
                )
            )
    return out


def run_ranking(settings: Settings, prefs: Preferences) -> RankingResult:
    return rank(candidates_from_db(settings), prefs)


def resolve_destination(
    settings: Settings,
    name: str,
    address: str,
    *,
    weight: float = 1.0,
    max_minutes: float | None = None,
    http: Http | None = None,
) -> tuple[Destination | None, str]:
    """Geocode a place the client named. Returns (destination, message-if-failed)."""
    with session_scope(settings) as s:
        found = Geocoder(settings, http).geocode_text(s, address)
    if found is None:
        return None, (f"Couldn't find “{address}”. Add the city and ZIP, or leave this place out.")
    return (
        Destination(
            name=name.strip() or "Place",
            lat=found[0],
            lon=found[1],
            weight=weight,
            max_minutes=max_minutes,
            address=address.strip(),
        ),
        "",
    )


def save_client(settings: Settings, label: str, prefs: Preferences) -> int:
    """Save a client's search criteria under a nickname (create or update)."""
    label = label.strip()
    if not label:
        raise ValueError("Give the client a nickname first.")
    with session_scope(settings) as s:
        client = s.scalar(select(Client).where(Client.label == label))
        if client is None:
            client = Client(label=label)
            s.add(client)
        client.preferences = prefs.to_dict()
        client.condition_mode = prefs.condition_mode
        s.flush()
        return client.id


def list_clients(settings: Settings) -> list[tuple[int, str]]:
    with session_scope(settings) as s:
        return [(c.id, c.label) for c in s.scalars(select(Client).order_by(Client.label))]


def load_client_prefs(settings: Settings, client_id: int) -> Preferences | None:
    with session_scope(settings) as s:
        client = s.get(Client, client_id)
        if client is None or not client.preferences:
            return None
        return Preferences.from_dict(client.preferences)


def first_photos(settings: Settings, listing_ids: list[int]) -> dict[int, Photo]:
    """First usable photo per listing, for thumbnails in result cards."""
    out: dict[int, Photo] = {}
    with session_scope(settings) as s:
        q = (
            select(Photo)
            .where(Photo.listing_id.in_(listing_ids), Photo.excluded.is_(False))
            .order_by(Photo.listing_id, Photo.order_idx)
        )
        for p in s.scalars(q):
            out.setdefault(p.listing_id, p)
    return out


def ranking_table(result: RankingResult) -> pd.DataFrame:
    """Flat table of the tier list (for display and CSV export)."""
    rows = []
    for r in result.ranked:
        c = r.candidate
        rows.append(
            {
                "Tier": r.tier,
                "Score": r.score,
                "Data coverage": f"{r.coverage:.0%}",
                "MLS #": c.mls or "",
                "Address": c.address,
                "Price": c.price,
                "Beds": c.beds,
                "Baths": c.baths,
                "Sq ft": c.sqft,
                "Strengths": " | ".join(r.strengths),
                "Watch-outs": " | ".join(r.watchouts),
                "Missing": " | ".join(r.unknowns),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ demo data and RentCast search

RENTCAST_CACHE_SECONDS = 6 * 3600


def load_demo_listings(settings: Settings, n: int = 36) -> ImportSummary:
    """Add clearly-labeled synthetic listings so anyone can try the app with no data."""
    summary = ImportSummary()
    with session_scope(settings) as s:
        batch = ImportBatch(source="demo", filename="demo listings", rows_total=n)
        s.add(batch)
        s.flush()
        res = upsert_listings(s, demo_listings(n), batch)
    summary.new, summary.updated, summary.listing_ids = res.new, res.updated, res.listing_ids
    return summary


@dataclass
class RentCastResult:
    summary: ImportSummary
    fetched: int
    from_cache: bool
    cache_age_minutes: int = 0
    skipped: int = 0


def _rentcast_cache_path(settings: Settings, params: dict) -> Path:
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
    return settings.data_dir / "cache" / f"rentcast_{key}.json"


def rentcast_search(
    settings: Settings,
    api_key: str,
    *,
    city: str | None = None,
    state: str | None = None,
    zip_code: str | None = None,
    max_price: int | None = None,
    property_types: list[str] | None = None,
    limit: int = 200,
    refresh: bool = False,
    http: RentCastHttp | None = None,
    now: float | None = None,
) -> RentCastResult:
    """Search RentCast (one API call) and add the results as listings.

    Results are cached for a few hours by search, so repeating a search doesn't spend more of the
    user's monthly quota. The API key is used only for the request: it is never written to the
    cache, the database or logs. The hosted site never supplies a key of its own.
    """
    src = RentCastSource(
        api_key,
        city=city,
        state=state,
        zip_code=zip_code,
        max_price=max_price,
        property_types=property_types or [],
        limit=limit,
        http=http,
    )
    params = src.request_params()  # validates the location before anything is spent
    path = _rentcast_cache_path(settings, params)
    now = time.time() if now is None else now
    payload, from_cache, age = None, False, 0
    if path.exists() and not refresh:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if now - cached["fetched_at"] < RENTCAST_CACHE_SECONDS:
                payload, from_cache = cached["payload"], True
                age = int((now - cached["fetched_at"]) // 60)
        except (OSError, ValueError, KeyError):
            payload = None
    if payload is None:
        payload = src.fetch_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": now, "payload": payload}), encoding="utf-8")

    raws = list(src.listings_from_payload(payload))
    summary = ImportSummary()
    with session_scope(settings) as s:
        batch = ImportBatch(source="rentcast", filename=json.dumps(params), rows_total=len(payload))
        s.add(batch)
        s.flush()
        res = upsert_listings(s, raws, batch)
        batch.rows_skipped = src.skipped
    summary.new, summary.updated, summary.listing_ids = res.new, res.updated, res.listing_ids
    return RentCastResult(summary, len(payload), from_cache, age, src.skipped)
