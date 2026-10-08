"""Runtime settings.

Secrets and machine paths come from the environment / `.env` (gitignored).
Non-secret preferences live in `<data_dir>/settings.yaml` so they can be edited in the UI.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOMINATIM_UA = "showing-planner/0.1 (local use; set NOMINATIM_USER_AGENT in .env)"

DISCLAIMER = (
    "Condition scores are an aid for the agent, not an appraisal or inspection. "
    "They describe visible physical condition in photos only."
)


@dataclass
class UserPrefs:
    """Editable, non-secret preferences persisted in settings.yaml."""

    default_city: str = ""
    default_state: str = "GA"
    service_minutes: int = 30
    buffer_before_minutes: int = 5
    buffer_after_minutes: int = 5

    @classmethod
    def load(cls, path: Path) -> UserPrefs:
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")


@dataclass
class Settings:
    data_dir: Path
    ors_api_key: str | None = None
    google_maps_api_key: str | None = None
    nominatim_user_agent: str = DEFAULT_NOMINATIM_UA
    prefs: UserPrefs = field(default_factory=UserPrefs)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def mappings_dir(self) -> Path:
        return self.data_dir / "mappings"

    @property
    def prefs_path(self) -> Path:
        return self.data_dir / "settings.yaml"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.photos_dir, self.mappings_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_prefs(self) -> None:
        self.prefs.save(self.prefs_path)

    @classmethod
    def from_env(cls, data_dir: Path | str | None = None, load_env_file: bool = True) -> Settings:
        if load_env_file:
            load_dotenv(PROJECT_ROOT / ".env")
        chosen = Path(data_dir or os.environ.get("SHOWING_DATA_DIR") or PROJECT_ROOT / "data")
        if not chosen.is_absolute():
            chosen = PROJECT_ROOT / chosen
        s = cls(
            data_dir=chosen,
            ors_api_key=os.environ.get("ORS_API_KEY") or None,
            google_maps_api_key=os.environ.get("GOOGLE_MAPS_API_KEY") or None,
            nominatim_user_agent=os.environ.get("NOMINATIM_USER_AGENT") or DEFAULT_NOMINATIM_UA,
        )
        s.prefs = UserPrefs.load(s.prefs_path)
        return s


def get_settings() -> Settings:
    return Settings.from_env()
