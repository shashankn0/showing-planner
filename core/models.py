"""SQLAlchemy models. Schema changes go through Alembic (core/migrations)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone type)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- listings


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))  # csv | manual | photo_folder | reso
    filename: Mapped[str | None] = mapped_column(String(512))
    profile_name: Mapped[str | None] = mapped_column(String(128))
    mapping_json: Mapped[dict | None] = mapped_column(JSON)
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_new: Mapped[int] = mapped_column(Integer, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Listing(Base):
    __tablename__ = "listings"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    source_listing_id: Mapped[str | None] = mapped_column(String(64), index=True)  # MLS number
    # "mls:<MLS#>" when an MLS number exists, otherwise "addr:<normalized address>"
    dedupe_key: Mapped[str] = mapped_column(String(300), unique=True)
    # Normalized street/unit/zip. Lets an address-only listing be enriched by a later MLS export.
    address_key: Mapped[str] = mapped_column(String(300), index=True, default="")

    street: Mapped[str | None] = mapped_column(String(200))
    unit: Mapped[str | None] = mapped_column(String(50))
    city: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(2))
    postal_code: Mapped[str | None] = mapped_column(String(10))
    address_full: Mapped[str] = mapped_column(String(400))
    county: Mapped[str | None] = mapped_column(String(100))

    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    geocode_status: Mapped[str] = mapped_column(String(16), default="none")
    # none | import | ok | failed | manual
    geocode_source: Mapped[str | None] = mapped_column(String(32))

    list_price: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(32))
    property_type: Mapped[str | None] = mapped_column(String(64))
    beds: Mapped[int | None] = mapped_column(Integer)
    baths_full: Mapped[int | None] = mapped_column(Integer)
    baths_half: Mapped[int | None] = mapped_column(Integer)
    baths_total: Mapped[float | None] = mapped_column(Float)
    sqft: Mapped[int | None] = mapped_column(Integer)
    lot_sqft: Mapped[int | None] = mapped_column(Integer)
    year_built: Mapped[int | None] = mapped_column(Integer)
    list_date: Mapped[date | None] = mapped_column(Date)

    # Showing instructions, with obvious access codes redacted at import.
    showing_requirements: Mapped[str | None] = mapped_column(Text)
    access_type: Mapped[str] = mapped_column(String(24), default="unknown")

    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    import_batch_id: Mapped[int | None] = mapped_column(ForeignKey("import_batches.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    photos: Mapped[list[Photo]] = relationship(
        back_populates="listing", cascade="all, delete-orphan", order_by="Photo.order_idx"
    )


class Photo(Base):
    __tablename__ = "photos"
    __table_args__ = (UniqueConstraint("listing_id", "sha256", name="uq_photo_listing_sha"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    # Relative to <data_dir>/photos when inside it, otherwise absolute (link mode).
    path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(32))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    order_idx: Mapped[int] = mapped_column(Integer, default=0)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    exclude_reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    listing: Mapped[Listing] = relationship(back_populates="photos")


# --------------------------------------------------------------------------- scoring


class PhotoScore(Base):
    __tablename__ = "photo_scores"
    __table_args__ = (UniqueConstraint("photo_id", "model_version", name="uq_photoscore"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    room_type: Mapped[str | None] = mapped_column(String(24))
    room_type_conf: Mapped[float | None] = mapped_column(Float)
    condition_score: Mapped[float | None] = mapped_column(Float)  # 1..5
    condition_conf: Mapped[float | None] = mapped_column(Float)  # 0..1
    flags: Mapped[dict | None] = mapped_column(JSON)  # {flag: probability}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ListingScore(Base):
    __tablename__ = "listing_scores"
    __table_args__ = (UniqueConstraint("listing_id", "model_version", name="uq_listingscore"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(
        ForeignKey("listings.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(64))
    overall: Mapped[float | None] = mapped_column(Float)
    worst_room: Mapped[str | None] = mapped_column(String(24))
    worst_room_score: Mapped[float | None] = mapped_column(Float)
    by_room: Mapped[dict | None] = mapped_column(JSON)
    weights: Mapped[dict | None] = mapped_column(JSON)
    n_photos_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"), index=True)
    condition: Mapped[int | None] = mapped_column(Integer)  # 1..5, null when `unsure`
    room_type: Mapped[str | None] = mapped_column(String(24))
    flags: Mapped[list | None] = mapped_column(JSON)
    unsure: Mapped[bool] = mapped_column(Boolean, default=False)
    rubric_version: Mapped[str] = mapped_column(String(16), default="v1")
    is_holdout: Mapped[bool] = mapped_column(Boolean, default=False)
    round_id: Mapped[int | None] = mapped_column(Integer)
    ms_spent: Mapped[int | None] = mapped_column(Integer)
    labeled_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- people


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    start_address: Mapped[str | None] = mapped_column(String(400))
    start_lat: Mapped[float | None] = mapped_column(Float)
    start_lon: Mapped[float | None] = mapped_column(Float)
    end_address: Mapped[str | None] = mapped_column(String(400))
    end_lat: Mapped[float | None] = mapped_column(Float)
    end_lon: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Client(Base):
    """Keep this minimal: a nickname is enough. Do not store sensitive personal data."""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"))
    label: Mapped[str] = mapped_column(String(120))
    # Stated preference chosen by the agent, never inferred: move_in_ready | renovation_ok | neutral
    condition_mode: Mapped[str] = mapped_column(String(24), default="neutral")
    # The client's stated search criteria (ranking.preferences.Preferences.to_dict()).
    preferences: Mapped[dict | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- routing


class Showing(Base):
    __tablename__ = "showings"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"))
    client_id: Mapped[int | None] = mapped_column(ForeignKey("clients.id"))
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    plan_date: Mapped[date | None] = mapped_column(Date)
    # The visit (arrival through departure) must fit inside [window_start, window_end].
    window_start: Mapped[time | None] = mapped_column(Time)
    window_end: Mapped[time | None] = mapped_column(Time)
    duration_min: Mapped[int] = mapped_column(Integer, default=30)
    buffer_before_min: Mapped[int] = mapped_column(Integer, default=5)
    buffer_after_min: Mapped[int] = mapped_column(Integer, default=5)
    status: Mapped[str] = mapped_column(String(16), default="requested")
    # requested | confirmed | denied
    access_type: Mapped[str | None] = mapped_column(String(24))
    client_rank: Mapped[int | None] = mapped_column(Integer)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"))
    plan_date: Mapped[date | None] = mapped_column(Date)
    provider: Mapped[str | None] = mapped_column(String(32))
    params: Mapped[dict | None] = mapped_column(JSON)
    total_drive_min: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    stops: Mapped[list[RouteStop]] = relationship(
        back_populates="route", cascade="all, delete-orphan", order_by="RouteStop.seq"
    )


class RouteStop(Base):
    __tablename__ = "route_stops"

    id: Mapped[int] = mapped_column(primary_key=True)
    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id", ondelete="CASCADE"), index=True)
    showing_id: Mapped[int | None] = mapped_column(ForeignKey("showings.id"))
    seq: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # start | showing | lunch | end
    arrival: Mapped[datetime | None] = mapped_column(DateTime)
    departure: Mapped[datetime | None] = mapped_column(DateTime)
    drive_min_from_prev: Mapped[float | None] = mapped_column(Float)
    wait_min: Mapped[float | None] = mapped_column(Float)
    dropped: Mapped[bool] = mapped_column(Boolean, default=False)
    drop_reason: Mapped[str | None] = mapped_column(Text)

    route: Mapped[Route] = relationship(back_populates="stops")


# --------------------------------------------------------------------------- caches


class TravelTimeCache(Base):
    __tablename__ = "travel_time_cache"
    __table_args__ = (
        UniqueConstraint("provider", "profile", "origin_key", "dest_key", name="uq_travel_cache"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(24))
    profile: Mapped[str] = mapped_column(String(24), default="free_flow")
    origin_key: Mapped[str] = mapped_column(String(40))  # rounded "lat,lon" key
    dest_key: Mapped[str] = mapped_column(String(40))
    duration_s: Mapped[float] = mapped_column(Float)
    distance_m: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)


class GeocodeCache(Base):
    __tablename__ = "geocode_cache"
    __table_args__ = (UniqueConstraint("provider", "query_key", name="uq_geocode_cache"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(24))  # census | nominatim
    query_key: Mapped[str] = mapped_column(String(400))  # normalized address
    status: Mapped[str] = mapped_column(String(16))  # ok | no_match | error
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    matched_address: Mapped[str | None] = mapped_column(String(400))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
