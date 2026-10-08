"""The importer must work on layouts it has never seen. These tests use several export styles."""

from pathlib import Path

import pytest

from core.config import UserPrefs
from sources.addresses import address_key, format_full, parse_full_address
from sources.csv_import import CsvImportSource, analyze_file
from sources.manual import ManualEntrySource
from sources.profiles import find_profile, save_profile
from sources.tabular import read_table
from sources.values import (
    classify_access,
    normalize_status,
    parse_baths,
    parse_price,
    redact_access_codes,
)

FIX = Path(__file__).parent / "fixtures"


def listings(name: str, **kw):
    a = analyze_file(FIX / name)
    src = CsvImportSource.from_analysis(a, **kw)
    return a, list(src.fetch()), src


# ------------------------------------------------------------------ layouts


def test_mls_style_layout_is_detected():
    a, rows, src = listings("synthetic_mls_style.csv")
    m = a.mapping
    assert m["source_listing_id"] == "MLS #"
    assert m["address"] == "Address"
    assert m["city"] == "City"
    assert m["state"] == "St"
    assert m["postal_code"] == "Zip"
    assert m["list_price"] == "List Price"
    assert m["beds"] == "Beds"
    assert m["baths_full"] == "Full Baths"
    assert m["baths_half"] == "Half Baths"
    assert m["sqft"] == "SqFt"
    assert m["year_built"] == "Year Built"
    assert m["status"] == "Status"
    assert m["showing_requirements"] == "Showing Instructions"
    # agent / office / photo columns must not be mistaken for listing fields
    assert not {"List Agent Name", "List Office", "Photo URL"} & set(m.values())
    assert any("photo URLs" in w for w in a.warnings)
    assert len(rows) == 6 and not src.skipped


def test_mls_style_values_are_normalized():
    _, rows, _ = listings("synthetic_mls_style.csv")
    r = rows[0]
    assert (r.source_listing_id, r.street, r.city, r.state, r.postal_code) == (
        "SYN7000101", "1001 Synthetic Way", "Testville", "GA", "30301",
    )  # fmt: skip
    assert r.list_price == 325_000 and r.beds == 3 and r.sqft == 1850 and r.year_built == 1998
    assert (r.baths_full, r.baths_half, r.baths_total) == (2, 1, 2.5)
    assert r.status == "active" and str(r.list_date) == "2026-09-12"
    assert rows[1].unit == "4B" and rows[1].street == "22 Sample Court NW"
    assert rows[2].status == "pending"
    assert rows[4].status == "coming_soon"
    assert rows[5].status == "pending"  # "Active Under Contract"


def test_access_codes_are_redacted_and_access_classified():
    _, rows, _ = listings("synthetic_mls_style.csv")
    assert "4821" not in rows[0].showing_requirements
    assert "9910" not in rows[3].showing_requirements
    assert "****" in rows[0].showing_requirements
    assert [r.access_type for r in rows[:3]] == ["appointment", "go_direct", "call_first"]


def test_reso_style_split_street_parts_are_composed():
    a, rows, _ = listings("synthetic_reso_style.csv")
    m = a.mapping
    assert m["street_number"] == "StreetNumber"
    assert m["street_name"] == "StreetName"
    assert m["street_suffix"] == "StreetSuffix"
    assert m["street_dir_prefix"] == "StreetDirPrefix"
    assert m["unit"] == "UnitNumber"
    assert m["source_listing_id"] == "ListingId"
    assert m["state"] == "StateOrProvince"
    assert m["baths_total"] == "BathroomsTotalInteger"
    assert m["sqft"] == "LivingArea"
    assert m["latitude"] == "Latitude" and m["longitude"] == "Longitude"
    assert m["county"] == "CountyOrParish"
    assert "ListAgentFullName" not in m.values()
    assert rows[0].street == "1450 Mockingbird Ln"
    assert rows[1].street == "88 N Madeup Ave" and rows[1].unit == "12"
    assert rows[0].latitude == pytest.approx(33.7701)
    assert rows[0].access_type == "appointment"


def test_portal_style_layout_and_combined_baths():
    a, rows, _ = listings("synthetic_portal_style.csv")
    m = a.mapping
    assert m["address"] == "ADDRESS"
    assert m["list_price"] == "PRICE"
    assert m["source_listing_id"] == "MLS#"
    assert m["baths_total"] == "BATHS"
    assert m["sqft"] == "SQUARE FEET"
    assert m["lot_sqft"] == "LOT SIZE"
    assert m["state"] == "STATE OR PROVINCE" and m["postal_code"] == "ZIP OR POSTAL CODE"
    assert m["latitude"] == "LATITUDE"
    # derived columns must not be taken for price or area
    assert "$/SQUARE FEET" not in m.values()
    assert "DAYS ON MARKET" not in m.values()
    assert rows[0].baths_total == 2.5 and (rows[0].baths_full, rows[0].baths_half) == (2, 1)
    assert rows[0].lot_sqft == 8712
    assert rows[2].unit == "1205" and rows[2].street == "77 Sample St"


def test_address_only_file_works_with_defaults():
    prefs = UserPrefs(default_city="Testville", default_state="GA")
    a, rows, src = listings("synthetic_addresses_only.csv", prefs=prefs)
    assert a.is_importable and list(a.mapping) == ["address"]
    assert len(rows) == 5
    assert rows[0].postal_code == "30301" and rows[0].city == "Testville"
    assert rows[1].unit == "4B" and rows[1].street == "22 Sample Court NW"
    assert rows[2].state == "GA"  # "Georgia" -> GA
    assert rows[4].city == "Testville" and rows[4].state == "GA"  # from defaults
    assert src.assumed_defaults >= 1


# ------------------------------------------------------------------ file oddities


def test_semicolon_cp1252_with_preamble_and_footer():
    text = (
        "Listing Report - exported 10/08/2026\n"
        "Prepared for: Test Agent\n"
        "\n"
        "Adresse;Ort;Staat;PLZ;Preis;Zimmer\n"  # deliberately foreign: must still import via content
        "12 Café Straße;Testville;GA;30301;$250,000;3\n"
        "14 Café Straße;Testville;GA;30301;$260,000;4\n"
        "2 listings exported\n"
    )
    t = read_table(text.encode("cp1252"), "report.csv")
    assert t.delimiter == ";" and t.encoding == "cp1252"
    assert t.header_row_index >= 1 and len(t.rows) == 2
    a = analyze_file(text.encode("cp1252"), "report.csv")
    assert a.is_importable
    src = CsvImportSource.from_analysis(a)
    got = list(src.fetch())
    assert [g.street for g in got] == ["12 Café Straße", "14 Café Straße"]
    assert got[0].state == "GA" and got[0].postal_code == "30301" and got[0].list_price == 250_000


def test_utf8_bom_and_tab_delimited():
    text = "﻿Address\tCity\tState\tZip\n5 Sample Rd\tTestville\tGA\t30301\n"
    a = analyze_file(text.encode("utf-8"), "x.tsv")
    got = list(CsvImportSource.from_analysis(a).fetch())
    assert got[0].street == "5 Sample Rd" and got[0].postal_code == "30301"


def test_headerless_file_is_inferred_from_values():
    text = (
        'SYN1100001,12 Sample Rd,Testville,GA,30301,"$250,000",33.7501,-84.3901\n'
        'SYN1100002,14 Sample Rd,Testville,GA,30301,"$260,000",33.7511,-84.3911\n'
        'SYN1100003,16 Sample Rd,Testville,GA,30302,"$270,000",33.7521,-84.3921\n'
    )
    a = analyze_file(text.encode(), "nohdr.csv")
    assert a.table.synthetic_headers
    m = a.mapping
    assert {"address", "state", "postal_code", "list_price", "latitude", "longitude"} <= set(m)
    got = list(CsvImportSource.from_analysis(a).fetch())
    assert got[0].street == "12 Sample Rd" and got[0].latitude == pytest.approx(33.7501)


def test_typo_and_odd_headers_fall_back_to_values_and_fuzzy_match():
    text = (
        "Prop Addr,Town,ST,Postal,Ask $,Bds,Bths,Sq Ft,Built,Ref #\n"
        '12 Sample Rd,Testville,GA,30301,"$250,000",3,2,"1,500",1999,SYN55001\n'
        '14 Sample Rd,Testville,GA,30301,"$260,000",4,3,"1,700",2004,SYN55002\n'
        '16 Sample Rd,Testville,GA,30302,"$270,000",3,2,"1,550",2011,SYN55003\n'
    )
    a = analyze_file(text.encode(), "odd.csv")
    m = a.mapping
    assert m["address"] == "Prop Addr"
    assert m["state"] == "ST" and m["postal_code"] == "Postal"
    assert m["list_price"] == "Ask $"
    assert m["beds"] == "Bds" and m["sqft"] == "Sq Ft" and m["year_built"] == "Built"
    assert m["source_listing_id"] == "Ref #"


def test_pasted_address_list_without_header():
    text = "12 Sample Rd, Testville, GA 30301\n14 Sample Rd, Testville, GA 30301\n"
    a = analyze_file(text.encode(), "list.txt")
    got = list(CsvImportSource.from_analysis(a).fetch())
    assert [g.street for g in got] == ["12 Sample Rd", "14 Sample Rd"]


def test_rows_without_address_are_skipped_with_reason():
    text = "Address,City\n12 Sample Rd,Testville\n,Testville\nN/A,Testville\n"
    a = analyze_file(text.encode(), "gaps.csv")
    src = CsvImportSource.from_analysis(a)
    got = list(src.fetch())
    assert len(got) == 1 and len(src.skipped) == 2


def test_unusable_file_reports_not_importable():
    text = "Color,Shape\nred,square\nblue,round\n"
    a = analyze_file(text.encode(), "nope.csv")
    assert not a.is_importable
    assert any("No address column" in w for w in a.warnings)


def test_user_override_wins_and_columns_are_not_reused():
    a = analyze_file(FIX / "synthetic_mls_style.csv", overrides={"beds": "Full Baths"})
    assert a.mapping["beds"] == "Full Baths"
    assert "baths_full" not in a.mapping  # the column moved to the new field


def test_xls_is_rejected_with_actionable_message():
    with pytest.raises(ValueError, match="Save As"):
        read_table(b"junk", "old.xls")


# ------------------------------------------------------------------ profiles


def test_saved_layout_is_recognized_next_time(tmp_path):
    a = analyze_file(FIX / "synthetic_mls_style.csv")
    save_profile(tmp_path, "My MLS export", a.mapping, a.table.headers)
    again = analyze_file(FIX / "synthetic_mls_style.csv", profiles_dir=tmp_path)
    assert again.profile_name == "My MLS export"
    assert again.mapping == a.mapping
    other = analyze_file(FIX / "synthetic_portal_style.csv", profiles_dir=tmp_path)
    assert other.profile_name is None
    assert find_profile(tmp_path, ["Totally", "Different"]) is None


# ------------------------------------------------------------------ manual + addresses


def test_manual_entry_parses_pasted_addresses():
    src = ManualEntrySource.from_text(
        "12 Sample Rd, Testville, GA 30301\n\n99 Mock Ave Apt 3, Demoburg, GA\n",
        UserPrefs(default_state="GA"),
    )
    got = list(src.fetch())
    assert len(got) == 2
    assert got[1].unit == "3" and got[1].street == "99 Mock Ave" and got[1].city == "Demoburg"


@pytest.mark.parametrize(
    ("text", "street", "city", "state", "zip_"),
    [
        ("123 Main St, Atlanta, GA 30301", "123 Main St", "Atlanta", "GA", "30301"),
        ("123 Main St, Atlanta, GA, 30301", "123 Main St", "Atlanta", "GA", "30301"),
        ("123 Main St Atlanta GA 30301", "123 Main St Atlanta", None, "GA", "30301"),
        ("123 Main St, Atlanta, Georgia 30301-1234", "123 Main St", "Atlanta", "GA", "30301"),
        ("123 Main St, Atlanta 30301", "123 Main St", "Atlanta", None, "30301"),
        ("123 Main St", "123 Main St", None, None, None),
    ],
)
def test_parse_full_address(text, street, city, state, zip_):
    p = parse_full_address(text)
    assert (p.street, p.city, p.state, p.postal_code) == (street, city, state, zip_)


def test_address_keys_match_across_formatting():
    a = address_key("123 Main Street", None, "Atlanta", "GA", "30301")
    b = address_key("123 MAIN ST.", None, "atlanta", "ga", "30301-9999")
    assert a == b
    assert a != address_key("123 Main Street", "4", "Atlanta", "GA", "30301")
    assert format_full("123 Main St", "4", "Atlanta", "GA", "30301") == (
        "123 Main St #4, Atlanta, GA 30301"
    )


# ------------------------------------------------------------------ value parsers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$425,000", 425_000), ("425000.00", 425_000), ("425K", 425_000), ("1.2M", 1_200_000),
     ("", None), ("N/A", None)],
)  # fmt: skip
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2.5", (2, 1, 2.5)), ("2/1", (2, 1, 2.5)), ("2 full 1 half", (2, 1, 2.5)),
     ("2F 1H", (2, 1, 2.5)), ("3", (3, 0, 3.0)), ("", (None, None, None))],
)  # fmt: skip
def test_parse_baths(raw, expected):
    assert parse_baths(raw) == expected


def test_status_and_access_helpers():
    assert normalize_status("Active Under Contract") == "pending"
    assert normalize_status("Closed") == "closed"
    assert classify_access("Courtesy call only") == "call_first"
    assert classify_access("") == "unknown"
    assert redact_access_codes("Gate code 1234#, call first") == "Gate code ****, call first"
    assert redact_access_codes("lockbox on door") == "lockbox on door"


# ------------------------------------------------------------------ unseen layouts (decoys)


def _analyze_rows(headers, rows):
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    return analyze_file(buf.getvalue().encode(), "t.csv")


def test_decoy_columns_are_not_mapped_in_a_wide_export():
    a = _analyze_rows(
        ["Street #", "Street Dir Prefix", "Street Name", "Street Suffix", "Unit #", "City",
         "State/Prov", "Postal Code", "List Price", "Original List Price", "Close Price",
         "Bedrooms", "Baths Total", "Approx Total SqFt", "Lot Size (Acres)", "Year Built",
         "Days on Market", "Listing Agent", "List Office Name", "MLS Number", "Tax Amount",
         "Parcel ID"],
        [["1001", "", "Synthetic", "Way", "", "Testville", "GA", "30301", "$325,000", "$335,000",
          "", "3", "2.5", "1,850", "0.25", "1998", "12", "Pat Example", "Demo Realty",
          "SYN7000101", "$2,100", "14 000 11 003"],
         ["22", "N", "Sample", "Ct", "4B", "Testville", "GA", "30302", "$215,500", "$215,500",
          "", "2", "2", "1,120", "0.05", "2004", "3", "Sam P", "Demo Realty", "SYN7000102",
          "$1,400", "14 000 11 004"],
         ["518", "", "Fakeland", "Dr", "", "Demoburg", "GA", "30303", "$489,900", "$499,000",
          "", "4", "3.5", "2,640", "0.4", "1987", "44", "Pat Example", "Sample Homes",
          "SYN7000103", "$3,900", "14 000 11 005"]],
    )  # fmt: skip
    m = a.mapping
    assert m["list_price"] == "List Price"
    assert m["lot_acres"] == "Lot Size (Acres)"
    assert m["sqft"] == "Approx Total SqFt"
    assert m["state"] == "State/Prov"
    assert "address" not in m  # Parcel ID must not be mistaken for an address
    assert not {"Original List Price", "Close Price", "Listing Agent", "List Office Name",
                "Tax Amount", "Parcel ID", "Days on Market"} & set(m.values())  # fmt: skip
    got = list(CsvImportSource.from_analysis(a).fetch())
    assert got[0].street == "1001 Synthetic Way" and got[0].lot_sqft == 10_890
    assert got[1].street == "22 N Sample Ct" and got[1].unit == "4B"


def test_short_ids_and_agent_contact_columns():
    a = _analyze_rows(
        ["Listing ID", "Property Address", "Mailing Address", "Listing Agent Email", "Agent Phone",
         "Listing Price", "Sold Price", "# Beds", "# Baths", "Living Area (SqFt)", "Yr Built"],
        [["SYN1", "1001 Synthetic Way, Testville, GA", "PO Box 12, Testville, GA 30301",
          "a@example.invalid", "555-0100", "$325,000", "", "3", "2", "1,850", "1998"],
         ["SYN2", "22 Sample Ct, Testville, GA", "PO Box 14, Testville, GA 30302",
          "b@example.invalid", "555-0101", "$215,500", "", "2", "2", "1,120", "2004"],
         ["SYN3", "518 Fakeland Dr, Demoburg, GA", "PO Box 15, Demoburg, GA 30303",
          "c@example.invalid", "555-0102", "$489,900", "$480,000", "4", "3", "2,640", "1987"]],
    )  # fmt: skip
    m = a.mapping
    assert m["source_listing_id"] == "Listing ID"
    assert m["address"] == "Property Address"
    assert m["beds"] == "# Beds" and m["baths_total"] == "# Baths"
    assert not {"Mailing Address", "Listing Agent Email", "Agent Phone", "Sold Price"} & set(
        m.values()
    )


def test_handmade_sheet_with_shorthand_prices():
    a = _analyze_rows(
        ["Home", "Where", "Notes", "Price", "Time"],
        [["A", "1001 Synthetic Way, Testville GA 30301", "lockbox", "325k", "1pm"],
         ["B", "22 Sample Ct, Testville GA 30302", "call owner", "215.5k", "2pm"],
         ["C", "518 Fakeland Dr, Demoburg GA 30303", "appt", "489.9k", "3pm"]],
    )  # fmt: skip
    assert a.mapping["address"] == "Where" and a.mapping["list_price"] == "Price"
    got = list(CsvImportSource.from_analysis(a).fetch())
    assert [g.list_price for g in got] == [325_000, 215_500, 489_900]
    assert got[0].state == "GA" and got[0].postal_code == "30301"


def test_junk_text_is_not_accepted_as_an_address():
    src = ManualEntrySource.from_text("hello\nN/A\n12 Sample Rd\nPeachtree Rd, Atlanta, GA\n")
    got = list(src.fetch())
    assert [g.street for g in got] == ["12 Sample Rd", "Peachtree Rd"]
    assert len(src.skipped) == 2
