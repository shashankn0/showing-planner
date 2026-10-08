"""Zero-shot baseline: judge photos by comparing them to text descriptions (no labels needed).

This is the starting point and the yardstick: a trained model has to beat it on your holdout.
CLIP-style models are good at "what is this a photo of" (room type, floor plan vs. kitchen) and
mediocre at subtle condition, so treat the zero-shot condition score as a rough first guess.

All prompts describe physical things (surfaces, damage, fixtures). None mentions people,
neighborhoods or price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.config import FLAGS, JUNK_LABELS, ROOMS
from vision.embedders import Embedder, VisionUnavailable

SCALE = 100.0  # CLIP's usual logit scale

ROOM_PROMPTS = {
    "kitchen": ["a photo of a kitchen", "a photo of a kitchen with cabinets and appliances"],
    "bath": ["a photo of a bathroom", "a photo of a bathroom with a toilet, sink or shower"],
    "living": ["a photo of a living room", "a photo of a family room with a sofa"],
    "bedroom": ["a photo of a bedroom", "a photo of a bedroom with a bed"],
    "exterior": ["a photo of the outside of a house", "a photo of a house exterior, yard or porch"],
    "other": [
        "a photo of a dining room, hallway, garage, basement, laundry room or office",
        "a photo of an empty room",
    ],
}
JUNK_PROMPTS = {
    "floor_plan": [
        "a floor plan drawing of a house",
        "an architectural blueprint with room labels",
    ],
    "map": ["a street map", "a satellite map with location pins"],
    "logo": ["a logo or graphic with text", "a real estate flyer or advertisement"],
    "aerial": ["an aerial drone photo of a neighborhood from high above"],
    "document": ["a scanned document or page of text"],
    "portrait": ["a headshot portrait photo"],
}
NOUN = {
    "kitchen": "kitchen",
    "bath": "bathroom",
    "living": "living room",
    "bedroom": "bedroom",
    "exterior": "house exterior",
    "other": "room",
}
LEVEL_PROMPTS = {
    5: [
        "a {n} that looks brand new or recently renovated with modern finishes",
        "a spotless, newly remodeled {n}",
        "a luxurious high-end {n} in perfect condition",
    ],
    4: [
        "a {n} in very good condition, clean and well maintained",
        "a tidy, well kept {n} with no visible problems",
    ],
    3: [
        "a {n} in average condition with normal wear and somewhat dated",
        "an ordinary, older but functional {n}",
    ],
    2: [
        "a {n} in fair condition with visible wear, damage or deferred maintenance",
        "a worn, outdated {n} that needs repairs",
    ],
    1: [
        "a {n} in very poor condition with severe damage, mold or disrepair",
        "a rundown, dilapidated, neglected {n}",
    ],
}
FLAG_PROMPTS = {
    "water_stain": (
        "a ceiling or wall with brown water stains",
        "a clean ceiling and walls with no stains",
    ),
    "damaged_flooring": (
        "damaged, scratched, stained or buckled flooring",
        "clean flooring in good condition",
    ),
    "dated_finishes": (
        "dated finishes from the 1980s or 1990s, oak cabinets, laminate counters, old fixtures",
        "modern, recently updated finishes and fixtures",
    ),
    "clutter": ("a cluttered messy room full of belongings", "a tidy, uncluttered room"),
    "poor_photo_quality": (
        "a blurry, dark or poorly lit photo",
        "a sharp, bright, professional real estate photo",
    ),
}


@dataclass
class ZeroShotOutput:
    room: str
    room_conf: float
    room_probs: dict[str, float]
    junk: str | None  # key of JUNK_LABELS when the photo isn't of the property
    junk_conf: float
    condition: float  # expected level 1..5
    condition_conf: float  # 1 - normalized entropy over the five levels
    flags: dict[str, float]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class ZeroShotModel:
    def __init__(self, embedder: Embedder) -> None:
        if not embedder.supports_text:
            raise VisionUnavailable(
                "Zero-shot scoring needs a CLIP-style backbone (set backbone: openclip:... in "
                "config/vision.yaml)."
            )
        self.embedder = embedder
        self._built = False

    def _mean_text(self, prompts: list[str]) -> np.ndarray:
        return _unit(self.embedder.embed_texts(prompts).mean(axis=0, keepdims=True))[0]

    def _build(self) -> None:
        if self._built:
            return
        self.junk_keys = list(JUNK_PROMPTS)
        self.class_matrix = np.stack(
            [self._mean_text(ROOM_PROMPTS[r]) for r in ROOMS]
            + [self._mean_text(JUNK_PROMPTS[j]) for j in self.junk_keys]
        )
        self.level_matrix = {
            room: np.stack(
                [
                    self._mean_text([t.format(n=NOUN[room]) for t in LEVEL_PROMPTS[lvl]])
                    for lvl in (1, 2, 3, 4, 5)
                ]
            )
            for room in ROOMS
        }
        self.flag_matrix = {
            f: np.stack([self._mean_text([pos]), self._mean_text([neg])])
            for f, (pos, neg) in FLAG_PROMPTS.items()
        }
        self._built = True

    def predict(self, embeddings: np.ndarray) -> list[ZeroShotOutput]:
        self._build()
        e = _unit(np.asarray(embeddings, dtype=np.float32))
        probs_all = _softmax(SCALE * e @ self.class_matrix.T)
        n_room = len(ROOMS)
        out = []
        for i in range(len(e)):
            p = probs_all[i]
            room_p = p[:n_room] / max(p[:n_room].sum(), 1e-9)
            room = ROOMS[int(room_p.argmax())]
            junk_p = p[n_room:]
            junk_total = float(junk_p.sum())
            junk = self.junk_keys[int(junk_p.argmax())] if junk_total > 0 else None
            lvl_p = _softmax(SCALE * e[i] @ self.level_matrix[room].T)
            expected = float((lvl_p * np.arange(1, 6)).sum())
            entropy = -float((lvl_p * np.log(np.maximum(lvl_p, 1e-12))).sum())
            flags = {f: float(_softmax(SCALE * e[i] @ self.flag_matrix[f].T)[0]) for f in FLAGS}
            out.append(
                ZeroShotOutput(
                    room=room,
                    room_conf=float(room_p.max()),
                    room_probs={r: float(x) for r, x in zip(ROOMS, room_p, strict=True)},
                    junk=junk,
                    junk_conf=junk_total,
                    condition=expected,
                    condition_conf=1.0 - entropy / float(np.log(5)),
                    flags=flags,
                )
            )
        return out

    def junk_label(self, key: str) -> str:
        return JUNK_LABELS.get(key, key)
