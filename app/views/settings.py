"""Settings: defaults, key status, saved layouts, and data-handling notes."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui import (
    DATA_NOTE,
    HOSTED_NOTE,
    auth_configured,
    context,
    disclaimer,
    signed_in_email,
    wipe_workspace,
)
from core.config import DEFAULT_NOMINATIM_UA
from sources.profiles import delete_profile, load_profiles


def _size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _photo_scoring_section(settings) -> None:  # noqa: ANN001
    from app import vision_services as vs
    from vision.config import ROOM_LABELS, ROOMS, VisionConfig, load_room_weights, save_room_weights
    from vision.embedders import deps_available
    from vision.pipeline import refresh_rollups

    st.subheader("Photo scoring")
    deps = deps_available()
    cfg = VisionConfig.load()
    active = vs.active_model_version(settings)
    ready = vs.vision_ready()
    st.dataframe(
        [
            {
                "Item": "Photo model software",
                "Status": "Installed"
                if ready
                else "Not installed (pip install -r requirements-vision.txt)",
            },
            {"Item": "Photo model", "Status": cfg.backbone},
            {"Item": "In use", "Status": active or "Zero-shot baseline (no trained model yet)"},
            {
                "Item": "Graphics card",
                "Status": "Not used: scoring runs on the CPU (plenty for this)"
                if not deps["torch"]
                else "CPU unless a CUDA build of PyTorch is installed",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Change the model in config/vision.yaml. Your labels and trained models stay on this computer."
    )

    st.markdown("**How much each room counts toward a listing's condition score**")
    weights = load_room_weights(settings.data_dir)
    cols = st.columns(len(ROOMS))
    new = {
        r: cols[i].number_input(
            ROOM_LABELS[r].split(" ")[0], 0.0, 10.0, float(weights[r]), step=0.5, key=f"rw_{r}"
        )
        for i, r in enumerate(ROOMS)
    }
    if st.button("Save room weights and recalculate scores"):
        save_room_weights(settings.data_dir, new)
        done = 0
        for v in {active, *[m["version"] for m in vs.model_versions(settings)]} - {None}:
            done += refresh_rollups(settings, v)
        for row in _zero_shot_versions(settings):
            done += refresh_rollups(settings, row)
        st.success(f"Saved. Recalculated {done} listing score(s).")


AUTH_EXAMPLE = Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml.example"


def _sign_in_section() -> None:
    st.subheader("Google sign-in")
    if auth_configured():
        email = signed_in_email()
        st.success(f"On. Signed in as {email}." if email else "On.")
        return
    st.caption(
        "Off. Visitors aren't asked to sign in. To require a Google sign-in, create an OAuth "
        "client (steps in docs/WEBSITE.md), then add this to `.streamlit/secrets.toml` or your "
        "host's Secrets and restart."
    )
    st.code(AUTH_EXAMPLE.read_text(encoding="utf-8"), language="toml")


def _zero_shot_versions(settings) -> list[str]:  # noqa: ANN001
    from sqlalchemy import select

    from core.db import session_scope
    from core.models import ListingScore

    with session_scope(settings) as s:
        return list(s.scalars(select(ListingScore.model_version).distinct()))


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

    _sign_in_section()

    _photo_scoring_section(settings)

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
