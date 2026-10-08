"""Plan a Day: placeholder until the routing phase."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import func, select

from app.ui import context, disclaimer
from core.db import session_scope
from core.models import Listing


def render() -> None:
    settings = context()
    st.title("Plan a Day")
    disclaimer()
    with session_scope(settings) as s:
        total = s.scalar(select(func.count()).select_from(Listing)) or 0
        mapped = (
            s.scalar(select(func.count()).select_from(Listing).where(Listing.latitude.is_not(None)))
            or 0
        )
    st.info(
        "Route planning arrives in a later phase: pick listings, set each showing window and "
        "duration, and get an ordered itinerary with drive times, a map, and .ics / printable "
        "exports. Nothing is ever sent to clients automatically."
    )
    c1, c2 = st.columns(2)
    c1.metric("Listings", total)
    c2.metric("With map coordinates", mapped)
    if total and mapped < total:
        st.caption(
            "Listings without coordinates can't be routed. Use the Listings page to look them "
            "up or place pins."
        )
