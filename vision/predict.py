"""Predictors: turn photo embeddings into room type, condition, flags (zero-shot or trained)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from vision.config import FLAGS, ROOMS, slug
from vision.embedders import Embedder
from vision.heads import (
    CornLinearHead,
    FlagClassifier,
    RidgeThresholdHead,
    RoomClassifier,
    make_features,
)
from vision.zeroshot import ZeroShotModel


@dataclass
class Prediction:
    room: str
    room_conf: float
    junk: str | None  # which non-property class this looks like, if any
    junk_conf: float
    condition: float  # 1..5
    condition_conf: float  # 0..1
    flags: dict[str, float]
    source: str  # "zero-shot" | "trained"


class Predictor(Protocol):
    version: str

    def predict(self, embeddings: np.ndarray) -> list[Prediction]: ...


class ZeroShotPredictor:
    source = "zero-shot"

    def __init__(self, embedder: Embedder, model: ZeroShotModel | None = None) -> None:
        self.model = model or ZeroShotModel(embedder)
        self.version = f"zeroshot-{slug(embedder.id)}-v1"

    def predict(self, embeddings: np.ndarray) -> list[Prediction]:
        return [
            Prediction(
                room=o.room,
                room_conf=o.room_conf,
                junk=o.junk,
                junk_conf=o.junk_conf,
                condition=o.condition,
                condition_conf=o.condition_conf,
                flags=o.flags,
                source=self.source,
            )
            for o in self.model.predict(embeddings)
        ]


@dataclass
class ModelBundle:
    version: str
    backbone_id: str
    condition_head: RidgeThresholdHead | CornLinearHead
    room_clf: RoomClassifier | None = None
    flag_clfs: dict[str, FlagClassifier] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    meta: dict = field(default_factory=dict)


class TrainedPredictor:
    source = "trained"

    def __init__(self, bundle: ModelBundle, zeroshot: ZeroShotModel | None = None) -> None:
        self.bundle = bundle
        self.zeroshot = zeroshot
        self.version = bundle.version

    def predict(self, embeddings: np.ndarray) -> list[Prediction]:
        n = len(embeddings)
        zs = self.zeroshot.predict(embeddings) if self.zeroshot else None
        if self.bundle.room_clf is not None:
            probs = self.bundle.room_clf.predict_proba(embeddings)
            rooms = [ROOMS[int(i)] for i in probs.argmax(axis=1)]
            room_conf = probs.max(axis=1)
        elif zs is not None:
            rooms, room_conf = [o.room for o in zs], np.array([o.room_conf for o in zs])
        else:
            rooms, room_conf = ["other"] * n, np.zeros(n)
        feats = make_features(embeddings, rooms)
        score = self.bundle.condition_head.predict_score(feats)
        conf = self.bundle.condition_head.confidence(feats)
        flag_vals = {}
        for f in FLAGS:
            if f in self.bundle.flag_clfs:
                flag_vals[f] = self.bundle.flag_clfs[f].predict_proba(embeddings)
            elif zs is not None:
                flag_vals[f] = np.array([o.flags[f] for o in zs])
            else:
                flag_vals[f] = np.zeros(n)
        return [
            Prediction(
                room=rooms[i],
                room_conf=float(room_conf[i]),
                junk=zs[i].junk if zs else None,
                junk_conf=zs[i].junk_conf if zs else 0.0,
                condition=float(score[i]),
                condition_conf=float(conf[i]),
                flags={f: float(flag_vals[f][i]) for f in FLAGS},
                source=self.source,
            )
            for i in range(n)
        ]
