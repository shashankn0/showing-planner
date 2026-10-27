"""Streamlit entry point:  streamlit run app/main.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui import account_box, load_settings, login_ok, mode_banner, password_ok  # noqa: E402
from app.views import (  # noqa: E402
    find_homes,
    labeling,
    listings,
    photo_review,
    plan_day,
    saved,
    settings,
)
from core.config import DISCLAIMER  # noqa: E402

st.set_page_config(page_title="Showing Planner", page_icon="🏠", layout="wide")

_settings = load_settings()
if not password_ok(_settings) or not login_ok():
    st.stop()

pages = [
    # the default page is served at the site root; a custom url_path there gives "Page not found"
    st.Page(find_homes.render, title="Find Homes", icon="🏆", default=True),
    st.Page(saved.render, title="Saved Homes", icon="⭐", url_path="saved"),
    st.Page(listings.render, title="Listings", icon="🏘️", url_path="listings"),
    st.Page(photo_review.render, title="Photo Review", icon="🖼️", url_path="photo-review"),
    st.Page(labeling.render, title="Labeling", icon="🏷️", url_path="labeling"),
    st.Page(plan_day.render, title="Plan a Day", icon="🗺️", url_path="plan-a-day"),
    st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
]
st.sidebar.caption(DISCLAIMER)
account_box()
mode_banner(_settings)
st.navigation(pages).run()
