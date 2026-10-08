"""Streamlit glue for photo scoring: load the heavy model once per server, not per page view."""

from __future__ import annotations

import streamlit as st

from app import vision_services as vs
from core.config import Settings
from vision.config import VisionConfig
from vision.embedders import make_embedder
from vision.runtime import Runtime, build_runtime, required_backbone
from vision.zeroshot import ZeroShotModel

INSTALL_HINT = "pip install -r requirements-vision.txt"


@st.cache_resource(
    show_spinner="Loading the photo model (the first time downloads about 600 MB)..."
)
def _embedder(backbone: str):  # noqa: ANN202
    return make_embedder(backbone)


@st.cache_resource(show_spinner=False)
def _zeroshot(backbone: str) -> ZeroShotModel | None:
    emb = _embedder(backbone)
    return ZeroShotModel(emb) if emb.supports_text else None


def get_runtime(settings: Settings) -> Runtime:
    """The model is shared by every session; the predictor and cache path are per workspace."""
    cfg = VisionConfig.load()
    backbone = required_backbone(settings, cfg)
    return build_runtime(settings, cfg, embedder=_embedder(backbone), zeroshot=_zeroshot(backbone))


def not_installed_notice() -> bool:
    """Show a friendly note and return True when the optional packages are missing."""
    if vs.vision_ready():
        return False
    st.info(
        "Photo scoring isn't enabled on this install. It needs optional software "
        f"(`{INSTALL_HINT}`). Everything else works without it; the tier list simply leaves "
        "photo condition out."
    )
    return True
