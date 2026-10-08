"""Remember column layouts the user has confirmed, so the next export of the same shape
imports with zero clicks. Stored as small YAML files under <data_dir>/mappings/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from sources.columns import normalize_header


def fingerprint(headers: list[str]) -> list[str]:
    return sorted({normalize_header(h) for h in headers if normalize_header(h)})


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "layout"


@dataclass
class Profile:
    name: str
    mapping: dict[str, str]
    headers: list[str]  # normalized fingerprint

    def similarity(self, headers: list[str]) -> float:
        a, b = set(self.headers), set(fingerprint(headers))
        return len(a & b) / len(a | b) if (a | b) else 0.0


def save_profile(directory: Path, name: str, mapping: dict[str, str], headers: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(name)}.yaml"
    path.write_text(
        yaml.safe_dump(
            {"name": name, "mapping": mapping, "headers": fingerprint(headers)}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return path


def load_profiles(directory: Path) -> list[Profile]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            out.append(Profile(raw["name"], raw["mapping"], raw["headers"]))
        except (KeyError, TypeError, yaml.YAMLError):
            continue  # ignore hand-edited files that no longer parse
    return out


def find_profile(
    directory: Path, headers: list[str], min_similarity: float = 0.85
) -> Profile | None:
    best, best_sim = None, 0.0
    cols = set(headers)
    for prof in load_profiles(directory):
        sim = prof.similarity(headers)
        # every mapped column must still exist under its exact header text
        if (
            sim >= min_similarity
            and sim > best_sim
            and all(c in cols for c in prof.mapping.values())
        ):
            best, best_sim = prof, sim
    return best


def delete_profile(directory: Path, name: str) -> bool:
    path = directory / f"{_slug(name)}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False
