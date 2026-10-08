"""Shared Streamlit helpers: app context, hosting safety, disclaimers, thumbnails."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from core.config import DISCLAIMER, Settings, UserPrefs, get_settings
from core.db import dispose_engine, init_db
from core.models import Photo
from sources.photos import resolve_photo_path

DATA_NOTE = (
    "Listing data and photos stay on this computer (the `data/` folder, never committed). "
    "Confirm with your broker what you may store and for how long."
)
HOSTED_NOTE = (
    "Shared website mode: your uploads live in a private, temporary workspace for this browser "
    "session, are deleted after about 2 idle hours, and you can delete them any time on the "
    "Settings page. Don't upload MLS-licensed data or photos here unless your broker has "
    "approved using a hosted tool."
)
THUMB_PX = 420
SESSION_ROOT = Path(tempfile.gettempdir()) / "showing_planner_sessions"
STALE_AFTER_SECONDS = 2 * 3600
SWEEP_EVERY_SECONDS = 600
MAX_PASSWORD_ATTEMPTS = 8  # per browser session
# Per-client throttle: a page reload gives a fresh session, so per-session limits alone don't stop
# brute force. Failures are counted per client address (as reported by the host's proxy), so one
# attacker locks only themselves out, not everyone.
CLIENT_MAX_FAILURES = 20
CLIENT_WINDOW_SECONDS = 900
CLIENT_LOCK_SECONDS = 900
MAX_TRACKED_CLIENTS = 5000  # bound memory if someone sprays many addresses

_last_sweep = 0.0
_failures: dict[str, list[float]] = {}
_locked_until: dict[str, float] = {}
_pause = time.sleep  # replaced in tests


def reset_auth_throttle() -> None:
    _failures.clear()
    _locked_until.clear()


LOOPBACK_ADDRESSES = {"127.0.0.1", "localhost", "::1"}


def server_is_loopback_only() -> bool:
    """True if Streamlit was started with --server.address bound to this machine only."""
    try:
        return str(st.get_option("server.address") or "").strip().lower() in LOOPBACK_ADDRESSES
    except Exception:  # noqa: BLE001
        return False


def proxy_trust_problem(settings: Settings) -> str | None:
    """Why TRUST_PROXY_HEADERS is switched on but being ignored (None = fine or not requested)."""
    if settings.trust_proxy_headers and not server_is_loopback_only():
        return (
            "TRUST_PROXY_HEADERS is on, but the app is listening on every network interface, so "
            "anyone who can reach it directly could forge client addresses. The setting is being "
            "ignored. Start the app with --server.address 127.0.0.1 so only your proxy can reach it."
        )
    return None


def _client_key(settings: Settings | None = None) -> str:
    """Identity used to throttle sign-in attempts.

    Client-address headers are believed only when the operator has said a trusted proxy sits in
    front (TRUST_PROXY_HEADERS=true) AND the server is bound to this machine only, so the proxy is
    the sole way in and nobody can forge the headers. Otherwise every visitor shares one key: that
    still blocks brute force, at the cost that a persistent attacker can lock the gate for
    everyone (use a private Community Cloud app or Cloudflare Access when that matters).
    """
    if not (settings and settings.trust_proxy_headers and server_is_loopback_only()):
        return "direct"
    try:
        headers = st.context.headers
        cf = headers.get("CF-Connecting-IP")
        if cf:
            return cf.strip()
        xff = headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[-1].strip()  # the hop our own proxy appended
    except Exception:  # noqa: BLE001
        pass
    return "direct"


def _lock_remaining(now: float, key: str) -> float:
    return max(_locked_until.get(key, 0.0) - now, 0.0)


def _note_failure(now: float, key: str) -> None:
    # forget expired state so the tables stay small
    for k in [k for k, t in _locked_until.items() if t <= now]:
        del _locked_until[k]
    for k in list(_failures):
        _failures[k] = [t for t in _failures[k] if now - t < CLIENT_WINDOW_SECONDS]
        if not _failures[k]:
            del _failures[k]
    # Spraying many fake addresses must not evict anyone's counts: once the table is full,
    # new addresses share a single bucket instead.
    bucket = key if (key in _failures or len(_failures) < MAX_TRACKED_CLIENTS) else "overflow"
    recent = [*_failures.get(bucket, []), now]
    _failures[bucket] = recent
    if len(recent) >= CLIENT_MAX_FAILURES:
        _locked_until[bucket] = now + CLIENT_LOCK_SECONDS
        _failures.pop(bucket, None)


def _sweep_stale_workspaces(now: float | None = None) -> None:
    """Delete visitor workspaces nobody has touched for a while (at most every 10 minutes)."""
    global _last_sweep  # noqa: PLW0603
    now = time.time() if now is None else now
    if now - _last_sweep < SWEEP_EVERY_SECONDS:
        return
    _last_sweep = now
    if not SESSION_ROOT.exists():
        return
    cutoff = now - STALE_AFTER_SECONDS
    for d in SESSION_ROOT.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                dispose_engine(f"sqlite:///{(d / 'app.db').as_posix()}")
        except OSError:
            continue


def wipe_workspace() -> None:
    """Delete this visitor's uploads and saved data right now (hosted mode)."""
    ws = st.session_state.pop("_workspace", None)
    st.session_state.pop("_db_ready", None)
    for key in ("ranking", "photo_scan"):
        st.session_state.pop(key, None)
    if ws:
        dispose_engine(f"sqlite:///{(Path(ws) / 'app.db').as_posix()}")
        shutil.rmtree(ws, ignore_errors=True)


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
    "TRUST_PROXY_HEADERS": "trust_proxy_headers",
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
        if attr in {"hosted", "trust_proxy_headers"}:
            on = str(value).strip().lower() in {"1", "true", "yes", "on"}
            setattr(settings, attr, getattr(settings, attr) or on)
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
    problem = proxy_trust_problem(settings)
    if problem:
        st.sidebar.warning(problem)
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
    st.title("Showing Planner")
    client = _client_key(settings)
    remaining = _lock_remaining(time.monotonic(), client)
    if remaining > 0:
        st.error(
            f"Too many failed sign-ins. Try again in about {math.ceil(remaining / 60)} minute(s)."
        )
        return False
    attempts = st.session_state.get("_attempts", 0)
    if attempts >= MAX_PASSWORD_ATTEMPTS:
        st.error("Too many attempts. Reload the page to try again.")
        return False
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Open"):
            if hmac.compare_digest(pw.encode(), settings.app_password.encode()):
                st.session_state["_authed"] = True
                st.rerun()
            _note_failure(time.monotonic(), client)
            _pause(min(0.25 * 2**attempts, 4.0))  # each wrong guess costs more time
            st.session_state["_attempts"] = attempts + 1
            if (
                attempts + 1 >= MAX_PASSWORD_ATTEMPTS
                or _lock_remaining(time.monotonic(), client) > 0
            ):
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
