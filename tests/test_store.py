from pathlib import Path

import pytest
from sqlalchemy import func, select

from core.config import Settings
from core.db import init_db, session_scope
from core.models import ImportBatch, Listing, Photo
from core.store import upsert_listings
from sources.base import RawListing
from sources.csv_import import CsvImportSource, analyze_file

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    init_db(s)
    return s


def raw(street="1 Test St", **kw):
    base = dict(source="manual", street=street, city="Testville", state="GA", postal_code="30301")
    base.update(kw)
    return RawListing(**base)


def test_migration_creates_all_tables(settings):
    from sqlalchemy import inspect

    from core.db import make_engine

    names = set(inspect(make_engine(settings.db_url)).get_table_names())
    expected = {
        "listings", "photos", "photo_scores", "listing_scores", "labels", "agents", "clients",
        "showings", "routes", "route_stops", "travel_time_cache", "geocode_cache",
        "import_batches",
    }  # fmt: skip
    assert expected <= names


def test_init_db_is_idempotent(settings):
    init_db(settings)
    init_db(settings)


def test_csv_import_then_reimport_updates_instead_of_duplicating(settings):
    a = analyze_file(FIX / "synthetic_mls_style.csv")
    with session_scope(settings) as s:
        batch = ImportBatch(source="csv", filename="x.csv")
        s.add(batch)
        s.flush()
        r1 = upsert_listings(s, CsvImportSource.from_analysis(a, is_synthetic=True).fetch(), batch)
    assert (r1.new, r1.updated) == (6, 0)
    with session_scope(settings) as s:
        r2 = upsert_listings(s, CsvImportSource.from_analysis(a, is_synthetic=True).fetch())
        assert (r2.new, r2.updated) == (0, 6)
        assert s.scalar(select(func.count()).select_from(Listing)) == 6
        assert all(x.is_synthetic for x in s.scalars(select(Listing)))


def test_address_only_listing_is_enriched_by_a_later_mls_import(settings):
    with session_scope(settings) as s:
        upsert_listings(s, [raw("1001 Synthetic Way")])
    with session_scope(settings) as s:
        res = upsert_listings(
            s,
            [raw("1001 Synthetic Way", source="csv", source_listing_id="SYN1", list_price=300_000)],
        )
        assert (res.new, res.updated) == (0, 1)
        lst = s.scalar(select(Listing))
        assert lst.source_listing_id == "SYN1" and lst.dedupe_key == "mls:SYN1"
        assert lst.list_price == 300_000


def test_same_address_without_mls_number_does_not_duplicate(settings):
    with session_scope(settings) as s:
        upsert_listings(s, [raw("1001 Synthetic Way", source_listing_id="SYN1")])
        res = upsert_listings(s, [raw("1001 SYNTHETIC WAY.")])
        assert (res.new, res.updated) == (0, 1)


def test_manual_pin_survives_reimport_with_coordinates(settings):
    with session_scope(settings) as s:
        upsert_listings(s, [raw(source_listing_id="SYN1", latitude=33.0, longitude=-84.0)])
        lst = s.scalar(select(Listing))
        assert lst.geocode_status == "import"
        lst.latitude, lst.longitude, lst.geocode_status = 33.5, -84.5, "manual"
    with session_scope(settings) as s:
        upsert_listings(s, [raw(source_listing_id="SYN1", latitude=1.0, longitude=2.0)])
        lst = s.scalar(select(Listing))
        assert (lst.latitude, lst.longitude, lst.geocode_status) == (33.5, -84.5, "manual")


def test_address_change_clears_stale_geocode(settings):
    with session_scope(settings) as s:
        upsert_listings(
            s, [raw("1 Test St", source_listing_id="SYN1", latitude=33.0, longitude=-84.0)]
        )
    with session_scope(settings) as s:
        upsert_listings(s, [raw("2 Other St", source_listing_id="SYN1")])
        lst = s.scalar(select(Listing))
        assert lst.latitude is None and lst.geocode_status == "none"


def test_deleting_a_listing_removes_its_photos(settings):
    with session_scope(settings) as s:
        upsert_listings(s, [raw()])
        lst = s.scalar(select(Listing))
        lst.photos.append(Photo(path="1/a.jpg", sha256="abc"))
    with session_scope(settings) as s:
        s.delete(s.scalar(select(Listing)))
    with session_scope(settings) as s:
        assert s.scalar(select(func.count()).select_from(Photo)) == 0
