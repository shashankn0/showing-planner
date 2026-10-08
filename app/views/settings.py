"""Settings: defaults, key status, saved layouts, and data-handling notes."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui import DATA_NOTE, HOSTED_NOTE, context, disclaimer, wipe_workspace
from core.config import DEFAULT_NOMINATIM_UA
from sources.profiles import delete_profile, load_profiles


def _size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def render() -> None:
    settings = context()
    st.title("Settings")
    disclaimer()

    st.subheader("Defaults")
    st.caption("Used when an import has no city/state, and as starting values when planning a day.")
    with st.form("prefs"):
        c1, c2 = st.columns(2)
        city = c1.text_input("Default city", value=settings.prefs.default_city)
        state = c2.text_input("Default state", value=settings.prefs.default_state, max_chars=2)
        c3, c4, c5 = st.columns(3)
        svc = c3.number_input(
            "Showing length (min)", 5, 240, settings.prefs.service_minutes, step=5
        )
        b1 = c4.number_input(
            "Buffer before (min)", 0, 90, settings.prefs.buffer_before_minutes, step=5
        )
        b2 = c5.number_input(
            "Buffer after (min)", 0, 90, settings.prefs.buffer_after_minutes, step=5
        )
        if st.form_submit_button("Save defaults"):
            p = settings.prefs
            p.default_city, p.default_state = city.strip(), state.strip().upper()
            p.service_minutes = int(svc)
            p.buffer_before_minutes, p.buffer_after_minutes = int(b1), int(b2)
            settings.save_prefs()
            st.success("Saved.")

    st.subheader("Services and keys")
    st.caption(
        "Keys live in the `.env` file next to the app (never committed). Edit it, then restart "
        "the app."
    )
    nominatim_ready = settings.nominatim_user_agent != DEFAULT_NOMINATIM_UA
    rows = [
        ("Census geocoder", "Ready: no key needed"),
        (
            "Fallback geocoder (Nominatim)",
            "Ready"
            if nominatim_ready
            else "Off: set NOMINATIM_USER_AGENT in .env to an identifying string with your contact info",
        ),
        (
            "OpenRouteService (drive times)",
            "Key found"
            if settings.ors_api_key
            else "No key: planning will use straight-line estimates",
        ),
        (
            "Google Maps (optional)",
            "Key found" if settings.google_maps_api_key else "Not set (optional)",
        ),
    ]
    st.dataframe([{"Service": a, "Status": b} for a, b in rows], hide_index=True, width="stretch")

    st.subheader("Saved column layouts")
    profiles = load_profiles(settings.mappings_dir)
    if not profiles:
        st.caption("None yet. Tick “Remember this layout” when importing a file.")
    for prof in profiles:
        a, b = st.columns([4, 1])
        a.write(f"**{prof.name}** · {len(prof.mapping)} fields mapped")
        if b.button("Forget", key=f"forget_{prof.name}"):
            delete_profile(settings.mappings_dir, prof.name)
            st.rerun()

    st.subheader("Your data")
    photos_mb = _size_mb(settings.photos_dir)
    if settings.hosted:
        st.write("Your uploads are in a private temporary workspace on the server.")
        st.info(HOSTED_NOTE)
        if st.button("Delete everything I uploaded now"):
            wipe_workspace()
            st.rerun()
    else:
        st.write(f"Location: `{settings.data_dir}`")
        st.info(DATA_NOTE)
    st.write(
        f"Photos: {photos_mb:.1f} MB · Everything else: {_size_mb(settings.data_dir) - photos_mb:.1f} MB"
    )
    st.markdown(
        "- This tool never scrapes MLS or listing sites.\n"
        "- Don't import alarm or lockbox codes. The importer masks digits after words like "
        "“code” or “alarm” as a safety net, but it isn't a guarantee.\n"
        "- Client records should be nicknames only."
    )
