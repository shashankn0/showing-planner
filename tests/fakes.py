"""Test doubles for the vision stack: fast, deterministic, no model downloads."""

import hashlib

import numpy as np
from PIL import Image

from vision.embedders import _l2
from vision.zeroshot import FLAG_PROMPTS, JUNK_PROMPTS, LEVEL_PROMPTS, NOUN, ROOM_PROMPTS


def text_vec(text: str, dim: int = 64) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=dim)


def class_vec(prompts: list[str]) -> np.ndarray:
    """What ZeroShotModel builds for a class: unit(mean(unit text vectors))."""
    v = _l2(np.stack([text_vec(p) for p in prompts])).mean(axis=0, keepdims=True)
    return _l2(v)[0]


class FakeClip:
    """Pretends to be a CLIP model. Text vectors are hash-random; images painted a 'marker' color
    get exactly the vector of a chosen class, so zero-shot picks it. Other images get a vector from
    their average color, so same-looking photos stay close together."""

    id = "fakeclip"
    supports_text = True

    def __init__(self) -> None:
        self.markers: dict[tuple[int, int, int], np.ndarray] = {}
        self.image_calls = 0

    def mark(self, color: tuple[int, int, int], prompts: list[str]) -> None:
        self.markers[color] = class_vec(prompts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([text_vec(t) for t in texts]))

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += len(images)
        out = []
        for im in images:
            px = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).reshape(-1, 3)
            mean = np.array(px.mean(axis=0))
            hit = next(
                (v for c, v in self.markers.items() if np.linalg.norm(mean - np.array(c)) <= 25),
                None,
            )
            if hit is not None and px.std(axis=0).max() < 45:
                out.append(hit)
            else:
                m = [int(x) // 16 for x in mean]
                out.append(text_vec(f"img{m[0]}-{m[1]}-{m[2]}"))
        return _l2(np.stack(out))


def kitchen_prompts() -> list[str]:
    return ROOM_PROMPTS["kitchen"]


def bath_prompts() -> list[str]:
    return ROOM_PROMPTS["bath"]


def floor_plan_prompts() -> list[str]:
    return JUNK_PROMPTS["floor_plan"]


__all__ = ["FLAG_PROMPTS", "LEVEL_PROMPTS", "NOUN", "FakeClip", "class_vec", "text_vec"]
