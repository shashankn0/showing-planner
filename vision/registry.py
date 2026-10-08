"""Where trained models live: <data_dir>/models/<version>/ (weights are never committed).

Each version folder holds
  bundle.json   structure and settings (plain JSON)
  bundle.npz    the numbers (plain arrays; loaded with allow_pickle=False)
  metrics.json  evaluation results
  model_card.md human-readable summary
ACTIVE.json names the version in use; if absent the zero-shot baseline is used.

No pickle anywhere: a model file can't run code when loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision.heads import FlagClassifier, RoomClassifier, condition_head_from_state
from vision.predict import ModelBundle


def _dir(models_dir: Path, version: str) -> Path:
    if not version or "/" in version or "\\" in version or version.startswith("."):
        raise ValueError("Invalid model version name.")
    return models_dir / version


def save_version(models_dir: Path, bundle: ModelBundle, metrics: dict, card_md: str) -> Path:
    d = _dir(models_dir, bundle.version)
    d.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    head_meta, head_arrays = bundle.condition_head.to_state()
    arrays.update({f"head/{k}": v for k, v in head_arrays.items() if v is not None})
    structure = {
        "version": bundle.version,
        "backbone_id": bundle.backbone_id,
        "created": bundle.created,
        "meta": bundle.meta,
        "head": head_meta,
        "room": None,
        "flags": {},
    }
    if bundle.room_clf is not None:
        structure["room"], room_arrays = bundle.room_clf.to_state()
        arrays.update({f"room/{k}": v for k, v in room_arrays.items()})
    for name, clf in bundle.flag_clfs.items():
        structure["flags"][name], flag_arrays = clf.to_state()
        arrays.update({f"flag/{name}/{k}": v for k, v in flag_arrays.items()})
    np.savez_compressed(d / "bundle.npz", **arrays)
    (d / "bundle.json").write_text(json.dumps(structure, indent=2, default=float), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (d / "model_card.md").write_text(card_md, encoding="utf-8")
    return d


def load_bundle(models_dir: Path, version: str) -> ModelBundle:
    d = _dir(models_dir, version)
    s = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    with np.load(d / "bundle.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    def sub(prefix: str) -> dict[str, np.ndarray]:
        return {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}

    head_arrays = sub("head/")
    for name in ("boot_coef", "boot_intercept", "const"):  # absent when never set
        head_arrays.setdefault(name, None)
    return ModelBundle(
        version=s["version"],
        backbone_id=s["backbone_id"],
        condition_head=condition_head_from_state(s["head"], head_arrays),
        room_clf=RoomClassifier.from_state(s["room"], sub("room/")) if s["room"] else None,
        flag_clfs={
            n: FlagClassifier.from_state(m, sub(f"flag/{n}/")) for n, m in s["flags"].items()
        },
        created=s["created"],
        meta=s["meta"],
    )


def list_versions(models_dir: Path) -> list[dict]:
    """Newest first. Each item is the version's metrics.json plus its name."""
    out = []
    if not models_dir.exists():
        return out
    for d in models_dir.iterdir():
        mj = d / "metrics.json"
        if d.is_dir() and mj.exists():
            try:
                out.append({"version": d.name, **json.loads(mj.read_text(encoding="utf-8"))})
            except (OSError, ValueError):
                continue
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


def active_version(models_dir: Path) -> str | None:
    f = models_dir / "ACTIVE.json"
    if not f.exists():
        return None
    try:
        version = json.loads(f.read_text(encoding="utf-8")).get("version")
        return version if version and (_dir(models_dir, version) / "bundle.npz").exists() else None
    except (OSError, ValueError):
        return None


def set_active(models_dir: Path, version: str | None) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    f = models_dir / "ACTIVE.json"
    if version is None:
        f.unlink(missing_ok=True)
        return
    if not (_dir(models_dir, version) / "bundle.npz").exists():
        raise ValueError(f"No saved model named '{version}'.")
    f.write_text(json.dumps({"version": version}), encoding="utf-8")
