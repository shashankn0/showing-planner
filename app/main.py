"""Streamlit entry point:  streamlit run app/main.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.views import labeling, listings, photo_review, plan_day, settings  # noqa: E402
from core.config import DISCLAIMER  # noqa: E402

st.set_page_config(page_title="Showing Planner", page_icon="🏠", layout="wide")

pages = [
    st.Page(listings.render, title="Listings", icon="🏘️", url_path="listings", default=True),
    st.Page(photo_review.render, title="Photo Review", icon="🖼️", url_path="photo-review"),
    st.Page(labeling.render, title="Labeling", icon="🏷️", url_path="labeling"),
    st.Page(plan_day.render, title="Plan a Day", icon="🗺️", url_path="plan-a-day"),
    st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
]
st.sidebar.caption(DISCLAIMER)
st.navigation(pages).run()
