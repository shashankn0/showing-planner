"""Shared Streamlit helpers: app context, hosting safety, disclaimers, thumbnails."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from core.config import DISCLAIMER, Settings, UserPrefs, get_settings
from core.db import dispose_engines, init_db
from core.models import Photo
from sources.photos import resolve_photo_path

DATA_NOTE = (
    "Listing data and photos stay on this computer (the `data/` folder, never committed). "
    "Confirm with your broker what you may store and for how long."
)
HOSTED_NOTE = (
    "Shared website mode: your uploads live in a private, temporary workspace for this browser "
    "session only and are deleted afterwards. Don't upload MLS-licensed data or photos here "
    "unless your broker has approved using a hosted tool."
)
THUMB_PX = 420
SESSION_ROOT = Path(tempfile.gettempdir()) / "showing_planner_sessions"
STALE_AFTER_SECONDS = 6 * 3600
MAX_PASSWORD_ATTEMPTS = 8

_swept = False


def _sweep_stale_workspaces() -> None:
    """Delete visitor workspaces nobody has touched for a while (once per server process)."""
    global _swept  # noqa: PLW0603
    if _swept or not SESSION_ROOT.exists():
        _swept = True
        return
    _swept = True
    cutoff = time.time() - STALE_AFTER_SECONDS
    for d in SESSION_ROOT.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue
    dispose_engines()


def _session_workspace() -> Path:
    """A private temp folder per browser session; unguessable name, kept alive while in use."""
    ws = st.session_state.get("_workspace")
    if not ws or not Path(ws).is_dir():
        _sweep_stale_workspaces()
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        ws = tempfile.mkdtemp(prefix="s_", dir=SESSION_ROOT)
        st.session_state["_workspace"] = ws
    os.utime(ws, None)  # mark as recently used
    return Path(ws)


_SECRET_FIELDS = {
    "SHOWING_HOSTED": "hosted",
    "APP_PASSWORD": "app_password",
    "ORS_API_KEY": "ors_api_key",
    "GOOGLE_MAPS_API_KEY": "google_maps_api_key",
    "NOMINATIM_USER_AGENT": "nominatim_user_agent",
}


def load_settings() -> Settings:
    """Environment settings, plus Streamlit secrets (how most free hosts pass configuration)."""
    settings = get_settings()
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001  (no secrets file locally; any failure means "none")
        return settings
    for key, attr in _SECRET_FIELDS.items():
        value = secrets.get(key)
        if value in (None, ""):
            continue
        if attr == "hosted":
            settings.hosted = settings.hosted or str(value).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        elif not getattr(settings, attr) or attr == "nominatim_user_agent":
            setattr(settings, attr, str(value))
    return settings


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", ""}


def is_exposed_host(host: str | None) -> bool:
    """True if the app is being served on an address other than this computer / home network."""
    h = (host or "").split(":")[0] if not (host or "").startswith("[") else "[::1]"
    h = h.strip().lower()
    if h in LOCAL_HOSTS or h.endswith(".local"):
        return False
    return not (h.startswith(("192.168.", "10.")) or h.startswith("172.") and _is_172_private(h))


def _is_172_private(host: str) -> bool:
    try:
        return 16 <= int(host.split(".")[1]) <= 31
    except (IndexError, ValueError):
        return False


def mode_banner(settings: Settings) -> None:
    """Sidebar note on which mode is running, and a loud warning for unsafe exposure."""
    try:
        host = st.context.headers.get("Host")
    except Exception:  # noqa: BLE001
        host = None
    if settings.hosted:
        st.sidebar.caption("Mode: shared website (temporary private workspaces)")
    elif is_exposed_host(host) and not settings.app_password:
        st.sidebar.error(
            "This app is reachable from the internet with no password, and everyone who opens it "
            "shares one set of saved data. Set APP_PASSWORD, or SHOWING_HOSTED=true for a public "
            "demo, then restart."
        )
    else:
        st.sidebar.caption("Mode: private (your data is saved on this computer)")


def context() -> Settings:
    """Settings for this session; creates data folders and migrates the database once.

    In hosted mode each visitor gets an isolated temporary workspace, so no one can see anyone
    else's listings or photos.
    """
    settings = load_settings()
    if settings.hosted:
        settings.data_dir = _session_workspace()
        settings.prefs = UserPrefs.load(settings.prefs_path)
        st.info(HOSTED_NOTE, icon="🔒")
    if st.session_state.get("_db_ready") != str(settings.data_dir):
        init_db(settings)
        st.session_state["_db_ready"] = str(settings.data_dir)
    return settings


def password_ok(settings: Settings) -> bool:
    """Optional shared-password gate (set APP_PASSWORD). True when the visitor may proceed."""
    if not settings.app_password:
        return True
    if st.session_state.get("_authed"):
        return True
    attempts = st.session_state.get("_attempts", 0)
    st.title("Showing Planner")
    if attempts >= MAX_PASSWORD_ATTEMPTS:
        st.error("Too many attempts. Reload the page to try again.")
        return False
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Open"):
            if hmac.compare_digest(pw.encode(), settings.app_password.encode()):
                st.session_state["_authed"] = True
                st.rerun()
            st.session_state["_attempts"] = attempts + 1
            if attempts + 1 >= MAX_PASSWORD_ATTEMPTS:
                st.rerun()  # show the lockout message right away
            st.error("That password isn't right.")
    return False


def disclaimer() -> None:
    st.caption(f"⚖️ {DISCLAIMER}")


def data_note() -> None:
    settings = load_settings()
    st.caption(f"🔒 {HOSTED_NOTE if settings.hosted else DATA_NOTE}")


def thumbnail(settings: Settings, photo: Photo, size: int = THUMB_PX) -> Path | None:
    """Cached, EXIF-rotated thumbnail so grids stay fast. Returns None if the file is gone."""
    src = resolve_photo_path(settings, photo.path)
    if not src.exists():
        return None
    out_dir = settings.data_dir / "thumbs"
    out = out_dir / f"{hashlib.sha1(photo.sha256.encode()).hexdigest()[:20]}_{size}.jpg"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            im.save(out, "JPEG", quality=85)
    return out
