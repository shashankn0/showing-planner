"""Engine, sessions, and migrations for the local SQLite database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.config import Settings, get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_engines: dict[str, Engine] = {}


def make_engine(url: str) -> Engine:
    if url in _engines:
        return _engines[url]
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    _engines[url] = engine
    return engine


def init_db(settings: Settings | None = None) -> Engine:
    """Create data dirs and bring the schema up to date. Safe to call repeatedly."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", settings.db_url.replace("%", "%%"))
    command.upgrade(cfg, "head")
    return make_engine(settings.db_url)


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    settings = settings or get_settings()
    session = sessionmaker(make_engine(settings.db_url), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engines() -> None:
    """Close pooled connections (needed on Windows before deleting a database file)."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()


def dispose_engine(url: str) -> None:
    """Close and forget the engine for one database (e.g. a deleted visitor workspace)."""
    engine = _engines.pop(url, None)
    if engine is not None:
        engine.dispose()
