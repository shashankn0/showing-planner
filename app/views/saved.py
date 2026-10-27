"""Saved Homes: the homes being tracked, where each one stands, and ways to look at them."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app import earth, services
from app.ui import context, disclaimer, md, thumbnail
from core.models import PIPELINE

LABELS = {"interested": "Interested", "toured": "Toured", "offer": "Offer", "passed": "Passed"}
VIEWER_HEIGHT = 520


def render() -> None:
    settings = context()
    st.title("Saved Homes")
    disclaimer()
    homes = services.saved_homes(settings)
    if not homes:
        st.info("Nothing saved yet. Build a tier list on Find Homes and press ☆ Save on a card.")
        return

    counts = {s: sum(h.status == s for h in homes) for s in PIPELINE}
    for col, s in zip(st.columns(len(PIPELINE)), PIPELINE, strict=True):
        col.metric(LABELS[s], counts[s])

    _viewer(settings, homes)

    photos = services.first_photos(settings, [h.listing_id for h in homes])
    tabs = st.tabs([f"{LABELS[s]} ({counts[s]})" for s in PIPELINE])
    for tab, s in zip(tabs, PIPELINE, strict=True):
        with tab:
            mine = [h for h in homes if h.status == s]
            if not mine:
                st.caption("Nothing here.")
            for h in mine:
                _card(settings, h, photos.get(h.listing_id))

    st.download_button(
        "Download for Google Earth (KML)",
        earth.kml(homes),
        file_name="saved_homes.kml",
        mime="application/vnd.google-earth.kml+xml",
        help="In Google Earth, choose File > Import KML file, then pick this file.",
    )


def _viewer(settings, homes: list[services.SavedItem]) -> None:  # noqa: ANN001
    st.subheader("See a home")
    by_id = {h.listing_id: h for h in homes}
    h = by_id[
        st.selectbox(
            "Home to view", list(by_id), format_func=lambda i: by_id[i].address, key="view_home"
        )
    ]
    placed = h.lat is not None and h.lon is not None
    c1, c2 = st.columns(2)
    c1.link_button("Open in Google Maps", earth.maps_url(h.address), width="stretch")
    if placed:
        c2.link_button("Satellite view", earth.satellite_url(h.lat, h.lon), width="stretch")
    else:
        st.caption(
            "This home has no map coordinates yet. Re-import it with “Look up map coordinates” on."
        )
    if not (placed and settings.google_maps_api_key):
        if placed:
            st.caption(
                "To see a 3D view inside this page, set GOOGLE_MAPS_API_KEY (see docs/WEBSITE.md)."
            )
        return
    # one switch per home, off by default: every 3D map load is billed by Google
    if st.toggle("Show 3D view", key=f"show3d_{h.listing_id}"):
        components.html(
            earth.viewer_html(h.lat, h.lon, settings.google_maps_api_key, VIEWER_HEIGHT),
            height=VIEWER_HEIGHT,
        )


def _set_status(settings, listing_id: int) -> None:  # noqa: ANN001
    services.update_saved(settings, listing_id, status=st.session_state[f"status_{listing_id}"])


def _set_notes(settings, listing_id: int) -> None:  # noqa: ANN001
    services.update_saved(settings, listing_id, notes=st.session_state[f"notes_{listing_id}"])


def _card(settings, h: services.SavedItem, photo) -> None:  # noqa: ANN001
    with st.container(border=True):
        left, right = st.columns([3, 4])
        with left:
            if photo is not None and (thumb := thumbnail(settings, photo, 360)):
                st.image(str(thumb), width="stretch")
            st.markdown(md(f"**{h.address}**"))
            facts = [
                f"${h.price:,}" if h.price else None,
                f"{h.beds} bd" if h.beds is not None else None,
                f"{h.baths:g} ba" if h.baths is not None else None,
                f"{h.sqft:,} sq ft" if h.sqft else None,
            ]
            st.caption(
                md(" · ".join(f for f in facts if f) + (" · demo data" if h.is_synthetic else ""))
            )
        with right:
            st.selectbox(
                "Status",
                PIPELINE,
                index=PIPELINE.index(h.status),
                format_func=LABELS.get,
                key=f"status_{h.listing_id}",
                on_change=_set_status,
                args=(settings, h.listing_id),
            )
            st.text_area(
                "Notes",
                value=h.notes,
                key=f"notes_{h.listing_id}",
                on_change=_set_notes,
                args=(settings, h.listing_id),
                height=90,
                placeholder="Showing times, what the client said. Nicknames only, nothing sensitive.",
            )
            st.button(
                "Remove from saved",
                key=f"rm_{h.listing_id}",
                on_click=services.unsave_home,
                args=(settings, h.listing_id),
            )
