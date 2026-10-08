"""Listings: import any export, add by address, attach photos, manage listings."""

from __future__ import annotations

import hashlib

import streamlit as st
from sqlalchemy import select

from app import services
from app.ui import context, data_note
from core.db import session_scope
from core.models import Listing
from sources.columns import FIELDS
from sources.csv_import import ImportAnalysis, analyze_file, analyze_table
from sources.profiles import load_profiles
from sources.rentcast import RentCastError

NONE = "(none)"
FIELD_GROUPS = [
    (
        "Address",
        ["address", "street_number", "street_dir_prefix", "street_name", "street_suffix",
         "street_dir_suffix", "unit", "city", "state", "postal_code"],
    ),
    ("Identity", ["source_listing_id", "status", "property_type", "list_date"]),
    ("Map position", ["latitude", "longitude", "county"]),
    (
        "Details",
        ["list_price", "beds", "baths_total", "baths_full", "baths_half", "sqft", "lot_sqft",
         "lot_acres", "year_built"],
    ),
    ("Showing", ["showing_requirements"]),
]  # fmt: skip
ACCESS_LABELS = {
    "go_direct": "Go direct (lockbox / vacant)",
    "call_first": "Call first",
    "appointment": "Appointment required",
    "unknown": "Unknown",
}


def render() -> None:
    settings = context()
    st.title("Listings")
    data_note()
    demo_button(settings, key="demo_listings_page")
    tab_import, tab_search, tab_add, tab_photos, tab_all = st.tabs(
        ["Import a file", "Search (RentCast)", "Add by address", "Photos", "All listings"]
    )
    with tab_import:
        _import_tab(settings)
    with tab_search:
        _search_tab(settings)
    with tab_add:
        _add_tab(settings)
    with tab_photos:
        _photos_tab(settings)
    with tab_all:
        _all_tab(settings)


def demo_button(settings, key: str) -> None:  # noqa: ANN001
    """One click to try the app with clearly-labeled synthetic listings."""
    if st.button(
        "Try it with demo listings", key=key, help="36 made-up listings, labeled as demo data."
    ):
        res = services.load_demo_listings(settings)
        st.success(
            f"Added {res.new} demo listings (synthetic: made-up addresses and numbers). "
            "Open Find Homes to build a tier list."
        )


# ------------------------------------------------------------------ search (RentCast)

PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Manufactured", "Multi-Family"]


def _search_tab(settings) -> None:  # noqa: ANN001
    st.write(
        "Search active for-sale listings by city or ZIP using **RentCast**, a licensed data "
        "service (no scraping). You use your own RentCast key."
    )
    with st.expander("What to know before searching"):
        st.markdown(
            "- Get a key at [rentcast.io/api](https://www.rentcast.io/api). Their free plan "
            "includes 50 calls a month; **each search here uses one call**, and repeating the same "
            "search within a few hours reuses the saved result instead of spending another.\n"
            "- **Your key stays in this browser session** and is sent only to RentCast. It is never "
            "saved, and this site never uses anyone else's key.\n"
            "- RentCast listings **don't include photos**, so photo-based condition can't be scored "
            "for them unless you add photos yourself.\n"
            "- Any charges on your RentCast account are yours."
        )
    env_key = None if settings.hosted else settings.rentcast_api_key
    typed = st.text_input(
        "RentCast API key",
        type="password",
        help="Leave empty to use the key from your .env file." if env_key else None,
    )
    c1, c2, c3 = st.columns([3, 1, 2])
    city = c1.text_input("City", key="rc_city", value=settings.prefs.default_city)
    state = c2.text_input("State", key="rc_state", value=settings.prefs.default_state, max_chars=2)
    zip_code = c3.text_input("…or ZIP code", key="rc_zip", max_chars=5)
    c1, c2, c3 = st.columns(3)
    max_price = c1.number_input(
        "Max price ($, optional)", 0, 20_000_000, 0, step=10_000, key="rc_max"
    )
    types = c2.multiselect("Property type (optional)", PROPERTY_TYPES, key="rc_types")
    limit = c3.slider("Most listings to fetch", 25, 500, 100, step=25, key="rc_limit")
    refresh = st.checkbox("Fetch fresh results even if I searched this recently (uses a call)")
    key = typed.strip() or env_key or ""
    if st.button("Search and add to my listings", type="primary"):
        try:
            with st.spinner("Searching RentCast…"):
                res = services.rentcast_search(
                    settings,
                    key,
                    city=city,
                    state=state,
                    zip_code=zip_code,
                    max_price=max_price or None,
                    property_types=types,
                    limit=limit,
                    refresh=refresh,
                )
        except RentCastError as exc:
            st.error(str(exc))
            return
        note = (
            f"reused a saved result from {res.cache_age_minutes} min ago"
            if res.from_cache
            else "used 1 API call"
        )
        st.success(
            f"{res.fetched} listings returned ({note}): {res.summary.new} new, "
            f"{res.summary.updated} updated. Open Find Homes to rank them."
        )
        if res.skipped:
            st.caption(f"{res.skipped} result(s) had no usable address and were skipped.")


# ------------------------------------------------------------------ import


def _import_tab(settings) -> None:  # noqa: ANN001
    st.write(
        "Upload **any** spreadsheet or export that has addresses: an MLS export, a portal "
        "download, a list you made yourself. The app works out which column is which; you "
        "confirm below. Only an address is required."
    )
    up = st.file_uploader("CSV, TSV, TXT or Excel (.xlsx) file", type=["csv", "tsv", "txt", "xlsx"])
    if up is None:
        return
    data = up.getvalue()
    fid = hashlib.md5(data).hexdigest()[:10]
    base_key = f"analysis_{fid}"
    if base_key not in st.session_state:
        try:
            st.session_state[base_key] = analyze_file(
                data, up.name, profiles_dir=settings.mappings_dir
            )
        except ValueError as exc:
            st.error(str(exc))
            return
    base: ImportAnalysis = st.session_state[base_key]
    if not base.table.rows:
        st.error("That file has no data rows. Check that it isn't empty.")
        return

    if base.profile_name:
        st.success(f"Recognized your saved layout “{base.profile_name}”.")
    for note in base.warnings:
        st.warning(note) if "No address" in note or "photo" in note.lower() else st.info(note)

    headers = base.table.headers
    selections: dict[str, str] = {}
    with st.expander(
        "Detected columns (change anything that's wrong)", expanded=not base.is_importable
    ):
        for title, names in FIELD_GROUPS:
            st.markdown(f"**{title}**")
            cols = st.columns(3)
            for i, fname in enumerate(names):
                current = base.mapping.get(fname, NONE)
                conf = base.detection.confidence.get(fname)
                hint = base.detection.reasons.get(fname, "")
                selections[fname] = cols[i % 3].selectbox(
                    FIELDS[fname].label,
                    [NONE, *headers],
                    index=([NONE, *headers].index(current) if current in headers else 0),
                    key=f"map_{fid}_{fname}",
                    help=(
                        f"Auto-detected ({conf:.0%}): {hint}" if conf and current != NONE else None
                    ),
                )
    overrides = {
        f: (None if v == NONE else v)
        for f, v in selections.items()
        if v != base.mapping.get(f, NONE)
    }
    analysis = analyze_table(base.table, settings.mappings_dir, overrides) if overrides else base
    if overrides and base.profile_name:  # a manual change means it's no longer the saved layout
        analysis.profile_name = None

    if not analysis.is_importable:
        st.error(
            "Pick the column that holds the street address (or both street number and street name) "
            "in “Detected columns” above."
        )
        return

    st.markdown(f"**Preview** (first rows of {len(base.table.rows):,})")
    preview = services.preview_rows(analysis, settings)
    st.dataframe(preview, hide_index=True, width="stretch")
    ignored = [h for h in headers if h not in analysis.mapping.values()]
    if ignored:
        st.caption(
            f"Not imported ({len(ignored)} columns): "
            + ", ".join(ignored[:12])
            + (" …" if len(ignored) > 12 else "")
        )

    c1, c2 = st.columns(2)
    with c1:
        save_layout = st.checkbox(
            "Remember this layout for next time", value=not analysis.profile_name
        )
        layout_name = st.text_input("Layout name", value="My export", disabled=not save_layout)
        demo = st.checkbox("These are demo / synthetic listings", value=False)
    with c2:
        geocode = st.checkbox("Look up map coordinates after importing", value=True)
        st.caption(
            "This sends street addresses to the free US Census geocoder to get coordinates. "
            "Uncheck to stay fully offline."
        )

    if st.button("Import listings", type="primary"):
        with st.spinner("Importing…"):
            summary = services.run_import(
                settings,
                analysis,
                up.name,
                is_synthetic=demo,
                save_layout_as=layout_name.strip() if save_layout and layout_name.strip() else None,
                geocode=geocode,
            )
        _show_summary(summary)


def _show_summary(summary: services.ImportSummary) -> None:
    st.success(f"Imported: {summary.new} new, {summary.updated} updated.")
    if summary.assumed_defaults:
        st.info(
            f"{summary.assumed_defaults} listing(s) had no city or state, so the defaults from "
            "Settings were assumed."
        )
    if summary.skipped:
        with st.expander(f"{len(summary.skipped)} row(s) skipped"):
            for n, why in summary.skipped[:50]:
                st.write(f"Row {n}: {why}")
    if summary.geocode:
        g = summary.geocode
        st.write(f"Map lookup: {g.ok} found, {g.failed} not found.")
        for m in dict.fromkeys(g.messages):
            st.caption(m)


# ------------------------------------------------------------------ add by address


def _add_tab(settings) -> None:  # noqa: ANN001
    st.write("Type or paste addresses, one per line. You can add details later.")
    text = st.text_area(
        "Addresses",
        height=160,
        placeholder="123 Main St, Atlanta, GA 30301\n45 Oak Ave Unit 3, Decatur, GA 30030",
    )
    geocode = st.checkbox("Look up map coordinates", value=True, key="add_geocode")
    if st.button("Add listings", disabled=not text.strip()):
        summary = services.add_addresses(settings, text, geocode=geocode)
        _show_summary(summary)


# ------------------------------------------------------------------ photos


def _photos_tab(settings) -> None:  # noqa: ANN001
    st.write(
        "Point to a folder (or upload a zip) of listing photos. Photos are matched to listings "
        "by **MLS number** or **address** found in the folder or file names. Nothing is "
        "downloaded from the internet."
    )
    with st.expander("Naming that works"):
        st.markdown(
            "- A folder per listing named by MLS number or address, e.g. `7000101/` or "
            "`1001 Synthetic Way/`\n"
            "- Loose files starting with the MLS number or address, e.g. `7000101_01.jpg`\n"
            "- Folder names can have extra words: `7000101 - Sample Way`\n"
            "- Anything that can't be matched, you assign by hand below."
        )
    # A hosted server can't see the visitor's folders, so only zip upload makes sense there.
    where = (
        "A zip file"
        if settings.hosted
        else st.radio(
            "Where are the photos?", ["A folder on this computer", "A zip file"], horizontal=True
        )
    )
    folder, zip_up = None, None
    if where.startswith("A folder"):
        folder = st.text_input("Folder path", placeholder=r"C:\Users\you\Documents\listing-photos")
    else:
        zip_up = st.file_uploader("Zip file of photos", type=["zip"])
    if settings.hosted:
        mode = "copy"
    else:
        mode_label = st.radio(
            "Keep the photos…",
            ["Copy into this app's data folder (recommended)", "Leave them where they are"],
            horizontal=True,
        )
        mode = "copy" if mode_label.startswith("Copy") else "link"

    if st.button("Scan photos", disabled=not (folder or zip_up)):
        try:
            st.session_state["photo_scan"] = services.scan_photos(
                settings,
                folder=None if zip_up else folder,
                zip_bytes=zip_up.getvalue() if zip_up else None,
            )
        except (ValueError, OSError) as exc:
            st.session_state.pop("photo_scan", None)
            st.error(str(exc))
    scan: services.PhotoScan | None = st.session_state.get("photo_scan")
    if scan is None:
        return
    _photo_assignment_ui(settings, scan, mode)


def _photo_assignment_ui(settings, scan: services.PhotoScan, mode: str) -> None:  # noqa: ANN001
    rep = scan.report
    n_photos = sum(len(g.files) for g in scan.groups)
    st.write(f"Found **{n_photos}** photos in **{len(scan.groups)}** set(s).")
    if not scan.groups:
        st.warning("No images found (looking for .jpg, .png, .webp, .bmp, .tif).")
        return

    assignments: dict[str, int | str | None] = {}
    if rep.matches:
        st.markdown("**Matched automatically**")
        st.dataframe(
            [
                {
                    "Set": m.group.key,
                    "Photos": len(m.group.files),
                    "Listing": scan.address_by_id.get(m.listing_id, ""),
                    "Matched by": "MLS number" if m.method == "mls_number" else "address",
                }
                for m in rep.matches
            ],
            hide_index=True,
            width="stretch",
        )
        for m in rep.matches:
            assignments[m.group.key] = m.listing_id

    needs_choice = [(g, None) for g in rep.unmatched] + list(rep.ambiguous)
    if needs_choice:
        st.markdown("**Needs your help**")
        ids = sorted(scan.address_by_id)
        special = {
            None: "Skip these photos",
            services.NEW_LISTING: "Create a new listing from this name",
        }

        def describe(option: int | str | None) -> str:
            return special.get(option) or scan.address_by_id.get(option, str(option))

        for g, cand_ids in needs_choice:
            options: list[int | str | None] = [None]
            if g.key in scan.new_listing_keys:
                options.append(services.NEW_LISTING)
            options += cand_ids if cand_ids else ids
            assignments[g.key] = st.selectbox(
                f"“{g.key}” ({len(g.files)} photos)",
                options,
                format_func=describe,
                key=f"assign_{g.key}",
                help="Ambiguous: more than one listing fits this name." if cand_ids else None,
            )

    n_assigned = sum(1 for v in assignments.values() if v is not None)
    if st.button(
        f"Import photos for {n_assigned} set(s)", type="primary", disabled=n_assigned == 0
    ):
        with st.spinner("Importing photos…"):
            res = services.apply_photos(settings, scan, assignments, mode)
        st.success(f"Added {res.added} photos ({res.duplicates} duplicates skipped).")
        if res.unreadable:
            st.warning("Couldn't read: " + ", ".join(res.unreadable[:10]))
        st.session_state.pop("photo_scan", None)


# ------------------------------------------------------------------ all listings


def _all_tab(settings) -> None:  # noqa: ANN001
    df = services.listings_table(settings)
    if df.empty:
        st.info("No listings yet. Import a file or add an address to get started.")
        return
    top = st.columns([3, 1, 1])
    query = top[0].text_input(
        "Search", placeholder="address, MLS #, status…", label_visibility="collapsed"
    )
    if query:
        mask = (
            df.astype(str)
            .apply(lambda col: col.str.contains(query, case=False, na=False))
            .any(axis=1)
        )
        df = df[mask]
    missing = int((df["Map"].isin(["none", "failed"])).sum())
    if top[1].button(f"Look up {missing} missing", disabled=missing == 0):
        rep = services.geocode_listing_ids(settings, force=True)
        st.toast(f"Map lookup: {rep.ok} found, {rep.failed} not found.")
        st.rerun()
    top[2].caption(f"{len(df):,} shown")

    event = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row"
    )
    rows = event.selection.rows if event and event.selection else []
    if not rows:
        st.caption("Select a row to edit it.")
        return
    _edit_panel(settings, int(df.iloc[rows[0]]["ID"]))


def _edit_panel(settings, listing_id: int) -> None:  # noqa: ANN001
    from core.geocoding import Geocoder

    with session_scope(settings) as s:
        lst = s.get(Listing, listing_id)
        if lst is None:
            return
        st.subheader(lst.address_full)
        if lst.is_synthetic:
            st.caption("Demo / synthetic listing")
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude", value=lst.latitude or 0.0, format="%.6f")
        lon = c2.number_input("Longitude", value=lst.longitude or 0.0, format="%.6f")
        c3.caption(
            f"Map status: {lst.geocode_status}"
            + (f" ({lst.geocode_source})" if lst.geocode_source else "")
        )
        keys = list(ACCESS_LABELS)
        access = st.selectbox(
            "How to get in",
            keys,
            index=keys.index(lst.access_type) if lst.access_type in keys else len(keys) - 1,
            format_func=ACCESS_LABELS.get,
            help="Follow the listing's own showing instructions; this only helps planning.",
        )
        if lst.showing_requirements:
            st.caption(f"Showing instructions on file: {lst.showing_requirements}")
        b1, b2, b3 = st.columns(3)
        if b1.button("Save pin and access"):
            if lat or lon:
                try:
                    Geocoder.set_manual_pin(lst, lat, lon)
                except ValueError as exc:
                    st.error(str(exc))
                    return
            lst.access_type = access
            st.toast("Saved.")
        confirm = b3.checkbox("Confirm delete")
        if b3.button("Delete listing", disabled=not confirm):
            s.delete(lst)
            st.rerun()
        _ = b2


def saved_layouts(settings) -> list:  # noqa: ANN001
    return load_profiles(settings.mappings_dir)


def listing_count(settings) -> int:  # noqa: ANN001
    with session_scope(settings) as s:
        return len(list(s.scalars(select(Listing.id))))
