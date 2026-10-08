import io
import zipfile

import pytest
from PIL import Image
from sqlalchemy import select

from core.config import Settings
from core.db import init_db, session_scope
from core.models import Listing
from core.store import upsert_listings
from sources.base import RawListing
from sources.photos import (
    extract_zip,
    ingest_photos,
    listings_from_groups,
    match_groups,
    resolve_photo_path,
    scan_photo_folder,
)


def make_img(path, color=(200, 30, 30), size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    init_db(s)
    return s


@pytest.fixture
def listings(settings):
    rows = [
        RawListing("csv", "1001 Synthetic Way", "Testville", "GA", "30301", source_listing_id="SYN7000101"),
        RawListing("csv", "22 Sample Ct NW", "Testville", "GA", "30302", unit="4B", source_listing_id="SYN7000102"),
        RawListing("csv", "22 Sample Ct NW", "Testville", "GA", "30302", unit="5A", source_listing_id="SYN7000109"),
        RawListing("csv", "518 Fakeland Dr", "Demoburg", "GA", "30303"),
    ]  # fmt: skip
    with session_scope(settings) as s:
        upsert_listings(s, rows)
        return list(s.scalars(select(Listing)))


def test_scan_groups_folders_and_loose_files(tmp_path):
    root = tmp_path / "in"
    make_img(root / "SYN7000101" / "a.jpg")
    make_img(root / "SYN7000101" / "sub" / "b.png")
    make_img(root / "loose_SYN7000102_01.jpg")
    make_img(root / "loose_SYN7000102_02.jpg")
    make_img(root / "loose_SYN7000102_10.jpg")
    (root / "notes.txt").write_text("not an image")
    groups = {g.key: g for g in scan_photo_folder(root)}
    assert set(groups) == {"SYN7000101", "loose_SYN7000102"}
    assert len(groups["SYN7000101"].files) == 2
    assert [f.name for f in groups["loose_SYN7000102"].files] == [
        "loose_SYN7000102_01.jpg", "loose_SYN7000102_02.jpg", "loose_SYN7000102_10.jpg",
    ]  # fmt: skip


def test_matching_by_mls_number_address_and_unit(tmp_path, listings):
    root = tmp_path / "in"
    make_img(root / "SYN7000101 - some notes" / "1.jpg")  # MLS number inside a longer name
    make_img(root / "518 Fakeland Dr" / "1.jpg")  # address folder (listing has no MLS#)
    make_img(root / "22_Sample_Court_NW_unit_5A" / "1.jpg")  # unit disambiguates
    make_img(root / "99 Unknown Rd" / "1.jpg")  # nothing to match
    report = match_groups(scan_photo_folder(root), listings)
    by_key = {m.group.key: m for m in report.matches}
    ids = {lst.source_listing_id or lst.street: lst.id for lst in listings}
    assert by_key["SYN7000101 - some notes"].listing_id == ids["SYN7000101"]
    assert by_key["SYN7000101 - some notes"].method == "mls_number"
    assert by_key["518 Fakeland Dr"].listing_id == ids["518 Fakeland Dr"]
    assert by_key["518 Fakeland Dr"].method == "address"
    assert by_key["22_Sample_Court_NW_unit_5A"].listing_id == ids["SYN7000109"]
    assert [g.key for g in report.unmatched] == ["99 Unknown Rd"]


def test_ambiguous_same_street_two_units_is_not_guessed(tmp_path, listings):
    root = tmp_path / "in"
    make_img(root / "22 Sample Court NW" / "1.jpg")
    report = match_groups(scan_photo_folder(root), listings)
    assert not report.matches and len(report.ambiguous) == 1


def test_ingest_copies_dedupes_and_flags_unreadable(tmp_path, settings, listings):
    src = tmp_path / "in"
    make_img(src / "a.jpg", (10, 10, 200))
    make_img(src / "b.jpg", (10, 200, 10))
    make_img(src / "b_copy.jpg", (10, 200, 10))  # identical pixels -> same bytes? re-save below
    (src / "b_copy.jpg").write_bytes((src / "b.jpg").read_bytes())
    (src / "broken.jpg").write_bytes(b"not really a jpeg")
    files = sorted(src.iterdir())
    with session_scope(settings) as s:
        lst = s.scalar(select(Listing).where(Listing.source_listing_id == "SYN7000101"))
        res = ingest_photos(s, settings, lst, files)
        assert (res.added, res.duplicates, res.unreadable) == (2, 1, ["broken.jpg"])
        again = ingest_photos(s, settings, lst, files)
        assert again.added == 0 and again.duplicates == 3
        paths = [resolve_photo_path(settings, p.path) for p in lst.photos]
        assert all(p.exists() and settings.photos_dir in p.parents for p in paths)
        assert lst.photos[0].width == 64 and lst.photos[0].height == 48
        assert [p.order_idx for p in lst.photos] == [0, 1]


def test_link_mode_references_files_in_place(tmp_path, settings, listings):
    src = tmp_path / "elsewhere"
    make_img(src / "a.jpg")
    with session_scope(settings) as s:
        lst = s.scalar(select(Listing).where(Listing.source_listing_id == "SYN7000101"))
        ingest_photos(s, settings, lst, [src / "a.jpg"], mode="link")
        assert resolve_photo_path(settings, lst.photos[0].path) == (src / "a.jpg").resolve()
        assert not list(settings.photos_dir.rglob("*.jpg"))


def test_photos_only_mode_creates_listings_from_address_folders(tmp_path):
    root = tmp_path / "in"
    make_img(root / "1001 Synthetic Way, Testville, GA 30301" / "1.jpg")
    make_img(root / "22_Sample_Ct" / "1.jpg")
    make_img(root / "IMG_0001.jpg")
    made, leftover = listings_from_groups(scan_photo_folder(root))
    streets = sorted(r.street for r in made)
    assert "1001 Synthetic Way" in streets
    assert any(r.city == "Testville" and r.postal_code == "30301" for r in made)
    assert [g.key for g in leftover] == ["IMG"]  # camera-named loose files stay unassigned


def test_zip_extraction_blocks_path_traversal(tmp_path):
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as zf:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        zf.writestr("SYN7000101/1.png", buf.getvalue())
    out = extract_zip(good.getvalue(), tmp_path / "x")
    assert (out / "SYN7000101" / "1.png").exists()

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        extract_zip(evil.getvalue(), tmp_path / "y")
    assert not (tmp_path / "escape.txt").exists()


def test_single_loose_file_named_by_mls_number_keeps_its_digits(tmp_path):
    root = tmp_path / "in"
    make_img(root / "SYN7000101.jpg")
    make_img(root / "SYN7000102 (2).jpg")
    make_img(root / "SYN7000102 (1).jpg")
    keys = {g.key: len(g.files) for g in scan_photo_folder(root)}
    assert keys == {"SYN7000101": 1, "SYN7000102": 2}
