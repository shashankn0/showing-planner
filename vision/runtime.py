"""Assemble the pieces for the current settings: which backbone, which predictor, which cache."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import Settings
from vision.config import VisionConfig
from vision.embedders import (
    Embedder,
    EmbeddingStore,
    VisionUnavailable,
    deps_available,
    make_embedder,
)
from vision.predict import ModelBundle, Predictor, TrainedPredictor, ZeroShotPredictor
from vision.registry import active_version, load_bundle
from vision.zeroshot import ZeroShotModel


@dataclass
class Runtime:
    cfg: VisionConfig
    embedder: Embedder
    store: EmbeddingStore
    predictor: Predictor
    zeroshot: ZeroShotModel | None
    bundle: ModelBundle | None
    notes: list[str] = field(default_factory=list)


def build_runtime(
    settings: Settings,
    cfg: VisionConfig | None = None,
    embedder: Embedder | None = None,
    zeroshot: ZeroShotModel | None = None,
) -> Runtime:
    """Use the active trained model if there is one (and its backbone), else the zero-shot baseline.

    `embedder` lets tests (or callers) supply their own backbone.
    """
    cfg = cfg or VisionConfig.load()
    notes: list[str] = []
    bundle: ModelBundle | None = None
    version = active_version(settings.models_dir)
    if version:
        try:
            bundle = load_bundle(settings.models_dir, version)
        except (OSError, ValueError, KeyError) as exc:
            notes.append(
                f"Couldn't load the saved model '{version}' ({type(exc).__name__}); using the baseline."
            )
    if bundle is not None:
        if embedder is None:
            embedder = make_embedder(bundle.backbone_id)
        elif embedder.id != bundle.backbone_id:
            notes.append(
                f"The saved model was trained on {bundle.backbone_id}, not {embedder.id}; using the baseline."
            )
            bundle = None
    if embedder is None:
        embedder = make_embedder(cfg.backbone)
    zs = (zeroshot or ZeroShotModel(embedder)) if embedder.supports_text else None
    if bundle is not None:
        predictor: Predictor = TrainedPredictor(bundle, zs)
    elif zs is not None:
        predictor = ZeroShotPredictor(embedder, zs)
    else:
        raise VisionUnavailable(
            "Scoring needs a CLIP-style backbone for the zero-shot baseline, or a model trained "
            "on your labels. Set backbone: openclip:... in config/vision.yaml."
        )
    return Runtime(
        cfg,
        embedder,
        EmbeddingStore(settings.embeddings_dir, embedder.id),
        predictor,
        zs,
        bundle,
        notes,
    )


def status(settings: Settings) -> dict:
    """Cheap facts for the UI (does not load any model)."""
    cfg = VisionConfig.load()
    return {
        "deps": deps_available(),
        "backbone": cfg.backbone,
        "active_model": active_version(settings.models_dir),
    }


def required_backbone(settings: Settings, cfg: VisionConfig | None = None) -> str:
    """The backbone scoring must use: the active trained model's, else the configured one."""
    import json

    cfg = cfg or VisionConfig.load()
    version = active_version(settings.models_dir)
    if version:
        try:
            meta = json.loads(
                (settings.models_dir / version / "bundle.json").read_text(encoding="utf-8")
            )
            return meta["backbone_id"]
        except (OSError, ValueError, KeyError):
            pass
    return cfg.backbone
