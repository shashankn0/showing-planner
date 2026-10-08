"""Labeling: placeholder until Phase 2."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import func, select

from app.ui import context, disclaimer
from core.db import session_scope
from core.models import Label, Photo


def render() -> None:
    settings = context()
    st.title("Labeling")
    disclaimer()
    with session_scope(settings) as s:
        n_photos = s.scalar(select(func.count()).select_from(Photo)) or 0
        n_labels = s.scalar(select(func.count()).select_from(Label)) or 0
    st.info(
        "The keyboard-driven labeling page arrives in the next phase. You'll rate photos 1-5 "
        "for condition with one keypress, tag flags (water stain, damaged flooring, dated "
        "finishes, clutter, poor photo), and the app will pick the most useful photos to show "
        "you next."
    )
    c1, c2 = st.columns(2)
    c1.metric("Photos available to label", n_photos)
    c2.metric("Photos labeled", n_labels)
    st.caption(
        "Aim for roughly 300-500 labeled photos across kitchens, baths, living areas, "
        "bedrooms and exteriors."
    )
