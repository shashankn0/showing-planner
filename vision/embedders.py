"""Turn photos (and prompts) into vectors.

Backends are loaded lazily so the rest of the app works without PyTorch installed.
Every embedding is cached on disk by the photo's content hash, so a photo is embedded once per
backbone no matter how many listings or runs include it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from vision.config import VisionConfig, slug
from vision.imaging import load_rgb

ProgressFn = Callable[[int, int], None]


class VisionUnavailable(RuntimeError):
    """Raised with a plain-language message when an optional dependency is missing."""


class Embedder(Protocol):
    id: str
    supports_text: bool

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def deps_available() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(mod) is not None
        for name, mod in {
            "torch": "torch",
            "open_clip": "open_clip",
            "timm": "timm",
            "scikit-learn": "sklearn",
        }.items()
    }


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-9)).astype(np.float32)


# ------------------------------------------------------------------ fallback (no download)


class ColorHistEmbedder:
    """A 64-number color histogram. Weak, but needs nothing: for tests and offline demos."""

    id = "colorhist"
    supports_text = False
    dim = 64

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for im in images:
            hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32)
            idx = (hsv // 64).clip(0, 3).astype(int)
            hist = np.zeros((4, 4, 4), dtype=np.float32)
            np.add.at(hist, (idx[..., 0], idx[..., 1], idx[..., 2]), 1.0)
            out.append(hist.flatten() / hist.sum())
        return _l2(np.stack(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise VisionUnavailable("The built-in fallback can't read text prompts.")


# ------------------------------------------------------------------ OpenCLIP


def _torch_device():  # noqa: ANN202
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class OpenClipEmbedder:
    """CLIP-style model: embeds photos and text prompts into one space (enables zero-shot)."""

    supports_text = True

    def __init__(self, model_name: str, pretrained: str) -> None:
        self.model_name, self.pretrained = model_name, pretrained
        self.id = f"openclip:{model_name}:{pretrained}"
        self._ready = False

    def _load(self) -> None:
        if self._ready:
            return
        if not deps_available()["open_clip"]:
            raise VisionUnavailable(
                "Photo scoring needs extra software that isn't installed here "
                "(pip install -r requirements-vision.txt)."
            )
        import open_clip

        self.device = _torch_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._ready = True

    @property
    def device_name(self) -> str:
        self._load()
        return self.device

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load()
        batch = torch.stack([self._preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
        return _l2(feats.float().cpu().numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        tokens = self._tokenizer(texts).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
        return _l2(feats.float().cpu().numpy())


class TimmEmbedder:
    """Vision-only backbone (e.g. DINOv2). No text prompts, so no zero-shot, but trainable."""

    supports_text = False

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.id = f"timm:{model_name}"
        self._ready = False

    def _load(self) -> None:
        if self._ready:
            return
        if not deps_available()["timm"]:
            raise VisionUnavailable(
                "This backbone needs extra software (pip install -r requirements-vision.txt)."
            )
        import timm

        self.device = _torch_device()
        self._model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self._model = self._model.eval().to(self.device)
        cfg = timm.data.resolve_data_config({}, model=self._model)
        self._tf = timm.data.create_transform(**cfg)
        self._ready = True

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load()
        batch = torch.stack([self._tf(im) for im in images]).to(self.device)
        with torch.no_grad():
            feats = self._model(batch)
        return _l2(feats.float().cpu().numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise VisionUnavailable("This backbone is vision-only and can't read text prompts.")


def make_embedder(backbone: str) -> Embedder:
    kind, _, rest = backbone.partition(":")
    if kind == "colorhist":
        return ColorHistEmbedder()
    if kind == "openclip":
        model, _, pretrained = rest.partition(":")
        if not (model and pretrained):
            raise ValueError("openclip backbones look like openclip:<model>:<pretrained>")
        return OpenClipEmbedder(model, pretrained)
    if kind == "timm":
        return TimmEmbedder(rest)
    raise ValueError(f"Unknown backbone '{backbone}'.")


# ------------------------------------------------------------------ cache


class EmbeddingStore:
    """One .npy per photo content hash, per backbone: <root>/<backbone>/<sha256>.npy"""

    def __init__(self, root: Path, backbone_id: str) -> None:
        self.dir = Path(root) / slug(backbone_id)

    def _path(self, sha: str) -> Path:
        return self.dir / f"{sha}.npy"

    def get(self, sha: str) -> np.ndarray | None:
        p = self._path(sha)
        if not p.exists():
            return None
        try:
            return np.load(p).astype(np.float32)
        except (OSError, ValueError):
            return None  # truncated file: recompute

    def put(self, sha: str, vec: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self._path(sha), vec.astype(np.float16))

    def get_many(self, shas: Iterable[str]) -> dict[str, np.ndarray]:
        out = {}
        for s in shas:
            v = self.get(s)
            if v is not None:
                out[s] = v
        return out


def embed_missing(
    store: EmbeddingStore,
    embedder: Embedder,
    items: list[tuple[str, Path]],
    cfg: VisionConfig,
    progress: ProgressFn | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return {sha: vector} for every item (computing only what isn't cached) and failed shas."""
    vectors: dict[str, np.ndarray] = {}
    todo: list[tuple[str, Path]] = []
    for sha, path in items:
        v = store.get(sha)
        if v is None:
            todo.append((sha, path))
        else:
            vectors[sha] = v
    failed: list[str] = []
    done = len(vectors)
    total = len(items)
    for i in range(0, len(todo), cfg.batch_size):
        chunk, imgs, ok = todo[i : i + cfg.batch_size], [], []
        for sha, path in chunk:
            try:
                imgs.append(load_rgb(path, cfg.max_side))
                ok.append(sha)
            except (OSError, ValueError):
                failed.append(sha)
        if imgs:
            for sha, vec in zip(ok, embedder.embed_images(imgs), strict=True):
                store.put(sha, vec)
                vectors[sha] = vec.astype(np.float16).astype(np.float32)  # same as a cache hit
        done += len(chunk)
        if progress:
            progress(min(done, total), total)
    return vectors, failed
