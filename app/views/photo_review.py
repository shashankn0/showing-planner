"""Photo Review: look through a listing's photos and exclude the ones that shouldn't be scored."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import select

from app.ui import context, disclaimer, thumbnail
from core.db import session_scope
from core.models import Listing

COLS = 4


def render() -> None:
    settings = context()
    st.title("Photo Review")
    disclaimer()
    with session_scope(settings) as s:
        listings = [
            (lst.id, lst.address_full, len(lst.photos))
            for lst in s.scalars(select(Listing).order_by(Listing.id.desc()))
        ]
    with_photos = [x for x in listings if x[2]]
    if not with_photos:
        st.info("No photos yet. Add some on the Listings page (Photos tab).")
        return
    labels = {i: f"{addr} ({n} photos)" for i, addr, n in with_photos}
    listing_id = st.selectbox("Listing", list(labels), format_func=labels.get)

    st.info(
        "Condition scores will appear here once the scoring model is added in the next phase. "
        "For now you can review photos and exclude any that shouldn't be scored "
        "(floor plans, maps, logos)."
    )
    with session_scope(settings) as s:
        lst = s.get(Listing, listing_id)
        cols = st.columns(COLS)
        for i, p in enumerate(lst.photos):
            with cols[i % COLS]:
                thumb = thumbnail(settings, p)
                if thumb is None:
                    st.warning("File missing")
                else:
                    st.image(str(thumb), width="stretch")
                excluded = st.checkbox("Exclude from scoring", value=p.excluded, key=f"ex_{p.id}")
                if excluded != p.excluded:
                    p.excluded = excluded
                    p.exclude_reason = "manual" if excluded else None
                st.caption(f"#{p.order_idx + 1} · {p.width}×{p.height}")
