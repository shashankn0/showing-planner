"""Shared Streamlit helpers: app context, disclaimers, thumbnails."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from core.config import DISCLAIMER, Settings, get_settings
from core.db import init_db
from core.models import Photo
from sources.photos import resolve_photo_path

DATA_NOTE = (
    "Listing data and photos stay on this computer (the `data/` folder, never committed). "
    "Confirm with your broker what you may store and for how long."
)
THUMB_PX = 420


def context() -> Settings:
    """Settings for this session; creates data folders and migrates the database once."""
    settings = get_settings()
    if st.session_state.get("_db_ready") != str(settings.data_dir):
        init_db(settings)
        st.session_state["_db_ready"] = str(settings.data_dir)
    return settings


def disclaimer() -> None:
    st.caption(f"⚖️ {DISCLAIMER}")


def data_note() -> None:
    st.caption(f"🔒 {DATA_NOTE}")


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
