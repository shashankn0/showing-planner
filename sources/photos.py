"""Photo intake: find photos in any folder/zip layout and match them to listings.

Layouts understood (mix and match):
  photos/<MLS#>/img1.jpg                       folder named by MLS number
  photos/1001 Synthetic Way/...                folder named by address
  photos/SYN7000101 - 1001 Synthetic Way/...   either, plus extra text
  photos/SYN7000101_01.jpg, SYN7000101-2.jpg   loose files prefixed by MLS number or address
  a .zip of any of the above

Nothing is downloaded from the internet. Photos are copied (or linked) into the local data dir.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import Settings
from core.models import Listing, Photo
from sources.addresses import clean, normalize_street, parse_full_address, street_tokens
from sources.base import RawListing

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_ZIP_FILES = 20_000
MAX_ZIP_BYTES = 4 * 1024**3

# Only strips an index that follows a separator ("_01", " (3)", "-2"), so MLS numbers survive.
_TRAILING_INDEX = re.compile(
    r"(?:[\s_\-.]+|\s*\()(?:photo|img|image|pic)?[\s_\-.]*\d{1,4}\)?$", re.I
)
_ADDRESS_START = re.compile(r"^\s*\d+[A-Za-z]?\s+[A-Za-z]")


@dataclass
class PhotoGroup:
    """Photos that belong together, labelled by the folder name or filename prefix."""

    key: str
    files: list[Path]


@dataclass
class Match:
    group: PhotoGroup
    listing_id: int
    method: str  # mls_number | address


@dataclass
class MatchReport:
    matches: list[Match] = field(default_factory=list)
    unmatched: list[PhotoGroup] = field(default_factory=list)
    ambiguous: list[tuple[PhotoGroup, list[int]]] = field(default_factory=list)


def natural_key(p: Path | str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def _images_under(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def scan_photo_folder(root: Path) -> list[PhotoGroup]:
    """Group images under `root` by subfolder, or by filename prefix for loose files."""
    root = Path(root)
    groups: list[PhotoGroup] = []
    loose: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir(), key=natural_key):
        if child.is_dir():
            files = _images_under(child)
            if files:
                groups.append(PhotoGroup(child.name, files))
        elif child.suffix.lower() in IMAGE_EXTS:
            key = _TRAILING_INDEX.sub("", child.stem).strip(" _-.") or child.stem
            loose.setdefault(key, []).append(child)
    groups += [PhotoGroup(k, sorted(v, key=natural_key)) for k, v in loose.items()]
    return groups


def extract_zip(zip_path_or_bytes: Path | bytes, dest: Path) -> Path:
    """Extract a photo zip with zip-slip and size protection. Returns the extraction folder."""
    import io

    src = (
        io.BytesIO(zip_path_or_bytes) if isinstance(zip_path_or_bytes, bytes) else zip_path_or_bytes
    )
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_FILES or sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
            raise ValueError(
                "That zip is too large to import in one go. Split it into smaller zips."
            )
        for info in infos:
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError("The zip contains unsafe file paths and was not extracted.")
        zf.extractall(dest)
    return dest


# ------------------------------------------------------------------ matching


def _alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def match_groups(groups: list[PhotoGroup], listings: list[Listing]) -> MatchReport:
    """Match each group to at most one listing: MLS number first, then address."""
    report = MatchReport()
    mls_index = {_alnum(lst.source_listing_id): lst for lst in listings if lst.source_listing_id}
    addr_index = [(lst, _match_tokens(lst)) for lst in listings]

    for g in groups:
        key_alnum = _alnum(g.key)
        hits = [lst for mls, lst in mls_index.items() if len(mls) >= 4 and mls in key_alnum]
        if len(hits) > 1:  # prefer the longest (most specific) MLS number contained in the key
            longest = max(len(_alnum(h.source_listing_id)) for h in hits)
            hits = [h for h in hits if len(_alnum(h.source_listing_id)) == longest]
        if len(hits) == 1:
            report.matches.append(Match(g, hits[0].id, "mls_number"))
            continue
        if len(hits) > 1:
            report.ambiguous.append((g, [h.id for h in hits]))
            continue

        key_tokens = street_tokens(re.sub(r"[_]+", " ", g.key))
        cands = [lst for lst, tok in addr_index if tok and tok <= key_tokens]
        if (
            len(cands) > 1
        ):  # several matched: prefer a unit named in the key, else the longest street
            with_unit = [c for c in cands if c.unit and c.unit.upper() in key_tokens]
            if with_unit:
                cands = with_unit
            else:
                best = max(len(_match_tokens(c)) for c in cands)
                cands = [c for c in cands if len(_match_tokens(c)) == best]
        if len(cands) == 1:
            report.matches.append(Match(g, cands[0].id, "address"))
        elif cands:
            report.ambiguous.append((g, [c.id for c in cands]))
        else:
            report.unmatched.append(g)
    return report


def _match_tokens(lst: Listing) -> set[str]:
    """House number + street words (suffix included) that a folder name must contain."""
    return street_tokens(lst.street or "")


def listings_from_groups(groups: list[PhotoGroup]) -> tuple[list[RawListing], list[PhotoGroup]]:
    """Photos-only mode: make listings from folders whose names look like addresses.

    Returns (listings, groups_that_could_not_be_turned_into_listings).
    """
    made, leftover = [], []
    for g in groups:
        name = clean(re.sub(r"[_]+", " ", g.key))
        parsed = parse_full_address(name)
        if _ADDRESS_START.match(name) and parsed.has_street:
            made.append(
                RawListing(
                    source="photo_folder",
                    street=parsed.street or "",
                    unit=parsed.unit,
                    city=parsed.city,
                    state=parsed.state,
                    postal_code=parsed.postal_code,
                )
            )
        else:
            leftover.append(g)
    return made, leftover


def group_for_listing(group: PhotoGroup, raw: RawListing) -> bool:
    """True if `group` is the folder that `listings_from_groups` derived `raw` from."""
    return normalize_street(clean(re.sub(r"[_]+", " ", group.key))).startswith(
        normalize_street(raw.street)
    )


# ------------------------------------------------------------------ ingest


@dataclass
class IngestResult:
    added: int = 0
    duplicates: int = 0
    unreadable: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_photos(
    session: Session,
    settings: Settings,
    listing: Listing,
    files: list[Path],
    mode: str = "copy",
) -> IngestResult:
    """Register `files` as photos of `listing`.

    mode="copy": copy into <data_dir>/photos/<listing id>/ (default; keeps everything local)
    mode="link": reference the files where they are (saves disk space; breaks if moved)
    """
    result = IngestResult()
    existing = {
        sha for sha in session.scalars(select(Photo.sha256).where(Photo.listing_id == listing.id))
    }
    next_idx = len(listing.photos)
    dest_dir = settings.photos_dir / str(listing.id)
    for src in sorted(files, key=natural_key):
        src = Path(src)
        try:
            with Image.open(src) as im:
                im.verify()
            with Image.open(src) as im:
                width, height = im.size
        except (UnidentifiedImageError, OSError, SyntaxError):
            result.unreadable.append(src.name)
            continue
        sha = _sha256(src)
        if sha in existing:
            result.duplicates += 1
            continue
        existing.add(sha)
        if mode == "copy":
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{next_idx + 1:03d}_{_safe_name(src.name)}"
            shutil.copy2(src, dest)
            stored = dest.relative_to(settings.photos_dir).as_posix()
        else:
            stored = str(src.resolve())
        listing.photos.append(
            Photo(path=stored, sha256=sha, width=width, height=height, order_idx=next_idx)
        )
        next_idx += 1
        result.added += 1
    session.flush()
    return result


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)


def resolve_photo_path(settings: Settings, stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else settings.photos_dir / p
