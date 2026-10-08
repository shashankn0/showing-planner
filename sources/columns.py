"""Layout-agnostic column detection.

Given any table (MLS export, portal download, hand-made spreadsheet), decide which column holds
which listing field. Evidence is combined from:

  1. the header text (synonyms from `config/column_aliases.yaml`, plus fuzzy matching), and
  2. the values (does this column *look like* zip codes / prices / US states / lat-lon ...).

Only a street address is required. Everything else is optional and the user can override any
decision in the UI. Nothing here knows about a particular MLS.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from sources.addresses import STATE_CODES, US_STATES
from sources.values import is_missing, to_float

ALIASES_PATH = Path(__file__).resolve().parent.parent / "config" / "column_aliases.yaml"
ACCEPT_THRESHOLD = 0.55
SAMPLE_SIZE = 300

# ------------------------------------------------------------------ header normalization


def normalize_header(header: object) -> str:
    """'ListPrice' / 'List_Price ($)' / 'LIST PRICE' -> 'list price'."""
    s = str(header or "")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)  # camelCase -> camel Case
    s = s.replace("#", " number ").replace("&", " and ").replace("$", " dollars ")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ content validators
# Each takes a sample of non-empty strings and returns the fraction that look plausible (0..1).


def _frac(values: Sequence[str], pred: Callable[[str], bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if pred(v)) / len(values)


def _v_zip(vs: Sequence[str]) -> float:
    return _frac(vs, lambda v: bool(re.fullmatch(r"\d{4,5}(-\d{4})?", v.strip())))


def _v_state(vs: Sequence[str]) -> float:
    return _frac(
        vs, lambda v: v.strip().upper().rstrip(".") in STATE_CODES or v.strip().upper() in US_STATES
    )


def _v_lat(vs: Sequence[str]) -> float:
    def ok(v: str) -> bool:
        f = to_float(v)
        return f is not None and 15 <= f <= 72 and "." in v

    return _frac(vs, ok)


def _v_lon(vs: Sequence[str]) -> float:
    def ok(v: str) -> bool:
        f = to_float(v)
        return f is not None and -180 <= f <= -60 and "." in v

    return _frac(vs, ok)


def _v_price(vs: Sequence[str]) -> float:
    from sources.values import parse_price

    def ok(v: str) -> bool:
        f = parse_price(v)
        return f is not None and 10_000 <= f <= 500_000_000

    return _frac(vs, ok)


def _v_small_int(lo: int, hi: int) -> Callable[[Sequence[str]], float]:
    def v(vs: Sequence[str]) -> float:
        def ok(x: str) -> bool:
            f = to_float(x)
            return f is not None and lo <= f <= hi

        return _frac(vs, ok)

    return v


def _v_sqft(vs: Sequence[str]) -> float:
    def ok(v: str) -> bool:
        f = to_float(v)
        return f is not None and 200 <= f <= 100_000

    return _frac(vs, ok)


def _v_year(vs: Sequence[str]) -> float:
    def ok(v: str) -> bool:
        f = to_float(v)
        return f is not None and 1700 <= f <= 2100 and float(f).is_integer()

    return _frac(vs, ok)


def _v_address_text(vs: Sequence[str]) -> float:
    """Looks like a street line: leading house number followed by words."""
    return _frac(vs, lambda v: bool(re.match(r"^\s*\d+[A-Za-z]?\s+[A-Za-z]{2,}", v)))


def _v_street_name(vs: Sequence[str]) -> float:
    return _frac(
        vs, lambda v: bool(re.match(r"^[A-Za-z]", v.strip())) and not re.search(r"\d{5}", v)
    )


def _v_house_number(vs: Sequence[str]) -> float:
    return _frac(vs, lambda v: bool(re.fullmatch(r"\d{1,6}[A-Za-z]?", v.strip())))


def _v_city(vs: Sequence[str]) -> float:
    return _frac(
        vs,
        lambda v: (
            bool(re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,40}", v.strip()))
            and v.strip().upper() not in STATE_CODES
        ),
    )


def _v_id(vs: Sequence[str]) -> float:
    """MLS-number-like: short alphanumeric tokens (no spaces) that are (almost) all unique."""
    if len(vs) < 2:
        return 0.5
    shape = _frac(vs, lambda v: bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,19}", v.strip())))
    uniq = len(set(vs)) / len(vs)
    return shape * (1.0 if uniq >= 0.9 else 0.3)


def _v_date(vs: Sequence[str]) -> float:
    from sources.values import parse_date

    return _frac(vs, lambda v: parse_date(v) is not None)


# ------------------------------------------------------------------ field specs


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    aliases: tuple[str, ...] = ()
    weak_aliases: tuple[str, ...] = ()
    reject_tokens: tuple[str, ...] = ()
    validator: Callable[[Sequence[str]], float] | None = None
    inferable: bool = False  # may be matched from content alone (no header evidence)
    min_content: float = 0.0  # below this, a header match is distrusted


_COMMON_REJECT = ("agent", "office", "broker", "email", "phone", "fax", "owner", "seller",
                  "url", "link", "photo", "image", "tour", "school", "tax", "legal")  # fmt: skip

FIELDS: dict[str, FieldSpec] = {
    f.name: f
    for f in (
        FieldSpec("source_listing_id", "MLS / listing number", validator=_v_id, min_content=0.5,
                  reject_tokens=("agent", "office", "broker", "photo", "url", "parcel", "tax")),
        FieldSpec("address", "Address (street, or full address)", validator=_v_address_text,
                  inferable=True, min_content=0.5, reject_tokens=_COMMON_REJECT + ("mailing",)),
        FieldSpec("street_number", "Street number", validator=_v_house_number, min_content=0.6,
                  reject_tokens=_COMMON_REJECT),
        FieldSpec("street_dir_prefix", "Street direction (prefix)", reject_tokens=_COMMON_REJECT),
        FieldSpec("street_name", "Street name", validator=_v_street_name, min_content=0.6,
                  reject_tokens=_COMMON_REJECT),
        FieldSpec("street_suffix", "Street suffix", reject_tokens=_COMMON_REJECT),
        FieldSpec("street_dir_suffix", "Street direction (suffix)", reject_tokens=_COMMON_REJECT),
        FieldSpec("unit", "Unit / apt", reject_tokens=_COMMON_REJECT),
        FieldSpec("city", "City", validator=_v_city, min_content=0.7,
                  reject_tokens=_COMMON_REJECT + ("mailing",)),
        FieldSpec("state", "State", validator=_v_state, inferable=True, min_content=0.8,
                  reject_tokens=_COMMON_REJECT),
        FieldSpec("postal_code", "ZIP", validator=_v_zip, inferable=True, min_content=0.8,
                  reject_tokens=_COMMON_REJECT + ("mailing",)),
        FieldSpec("county", "County", reject_tokens=_COMMON_REJECT),
        FieldSpec("latitude", "Latitude", validator=_v_lat, inferable=True, min_content=0.8,
                  reject_tokens=_COMMON_REJECT),
        FieldSpec("longitude", "Longitude", validator=_v_lon, inferable=True, min_content=0.8,
                  reject_tokens=_COMMON_REJECT),
        FieldSpec("list_price", "List price", validator=_v_price, inferable=True, min_content=0.6,
                  reject_tokens=("original", "previous", "prior", "per", "tax", "hoa", "assoc",
                                 "sqft", "sq ft", "square", "close", "sold", "rent", "lease", "fee")),
        FieldSpec("status", "Status", reject_tokens=("agent", "office", "change", "date")),
        FieldSpec("property_type", "Property type", reject_tokens=("agent", "office", "photo")),
        FieldSpec("beds", "Beds", validator=_v_small_int(0, 30), min_content=0.8,
                  reject_tokens=("sqft", "price", "agent")),
        FieldSpec("baths_total", "Baths (total)", validator=_v_small_int(0, 30), min_content=0.5,
                  reject_tokens=("full", "half", "partial", "quarter", "sqft", "price")),
        FieldSpec("baths_full", "Full baths", validator=_v_small_int(0, 30), min_content=0.6),
        FieldSpec("baths_half", "Half baths", validator=_v_small_int(0, 30), min_content=0.6),
        FieldSpec("sqft", "Living area (sq ft)", validator=_v_sqft, min_content=0.6,
                  reject_tokens=("lot", "acre", "price", "per", "dollars", "garage", "basement",
                                 "porch", "deck", "unfinished")),
        FieldSpec("lot_sqft", "Lot size (sq ft)", reject_tokens=("acre", "price")),
        FieldSpec("lot_acres", "Lot size (acres)", reject_tokens=("price",)),
        FieldSpec("year_built", "Year built", validator=_v_year, min_content=0.6,
                  reject_tokens=("renov", "remodel", "update", "tax")),
        FieldSpec("list_date", "List date", validator=_v_date, min_content=0.5,
                  reject_tokens=("expir", "close", "sold", "status", "change", "modif", "update",
                                 "contract close", "pending")),
        FieldSpec("showing_requirements", "Showing instructions", reject_tokens=("agent phone",)),
    )
}  # fmt: skip


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ------------------------------------------------------------------ detection


@dataclass
class ColumnMatch:
    field: str
    column: str
    confidence: float
    reason: str


@dataclass
class MappingResult:
    mapping: dict[str, str] = field(default_factory=dict)  # canonical field -> column name
    confidence: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_address(self) -> bool:
        m = self.mapping
        return "address" in m or ("street_name" in m and "street_number" in m)

    @property
    def is_importable(self) -> bool:
        return self.has_address


def _name_score(
    spec: FieldSpec, header_norm: str, aliases: dict[str, list[str]]
) -> tuple[float, str]:
    if not header_norm:
        return 0.0, ""
    tokens = set(header_norm.split())
    for rej in spec.reject_tokens:
        if rej in header_norm:
            return 0.0, f"header contains '{rej}'"
    strong = [normalize_header(a) for a in aliases.get("aliases", [])] + [
        normalize_header(a) for a in spec.aliases
    ]
    weak = [normalize_header(a) for a in aliases.get("weak_aliases", [])] + [
        normalize_header(a) for a in spec.weak_aliases
    ]
    if header_norm in strong:
        return 1.0, "header matches a known name"
    if header_norm in weak:
        return 0.7, "header resembles a known name"
    best, why = 0.0, ""
    for alias_list, cap in ((strong, 0.9), (weak, 0.65)):
        for a in alias_list:
            if not a:
                continue
            a_tokens = set(a.split())
            if len(a) >= 4 and a_tokens <= tokens:  # all alias words present in header
                score = cap * (0.9 if len(tokens) > len(a_tokens) else 1.0)
                if score > best:
                    best, why = score, f"header contains '{a}'"
            elif len(a) >= 3:
                ratio = SequenceMatcher(None, header_norm, a).ratio()
                if ratio >= 0.84 and ratio * cap > best:
                    best, why = ratio * cap, f"header is close to '{a}'"
    return best, why


def detect_mapping(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aliases: dict[str, dict[str, list[str]]] | None = None,
) -> MappingResult:
    """Infer {canonical field -> column header} from headers and sample values."""
    aliases = aliases if aliases is not None else load_aliases()
    n_cols = len(headers)
    sample_rows = rows[: SAMPLE_SIZE * 3]
    samples: list[list[str]] = []
    for c in range(n_cols):
        col = [r[c].strip() for r in sample_rows if c < len(r) and not is_missing(r[c])]
        samples.append(col[:SAMPLE_SIZE])

    candidates: list[tuple[float, str, int, str]] = []  # (score, field, col_idx, reason)
    for ci, header in enumerate(headers):
        h_norm = normalize_header(header)
        values = samples[ci]
        if not values:
            continue
        for spec in FIELDS.values():
            name_score, name_reason = _name_score(spec, h_norm, aliases.get(spec.name, {}))
            content = spec.validator(values) if spec.validator else None
            if name_score > 0:
                if content is None:
                    score = name_score * 0.95
                    reason = name_reason
                elif content >= spec.min_content:
                    score = name_score * 0.65 + content * 0.35
                    reason = f"{name_reason}; {content:.0%} of values fit"
                else:
                    score = name_score * 0.35  # header says yes, values say no
                    reason = f"{name_reason}, but only {content:.0%} of values fit"
                candidates.append((score, spec.name, ci, reason))
            elif spec.inferable and content is not None and content >= max(0.9, spec.min_content):
                candidates.append((content * 0.6, spec.name, ci, f"{content:.0%} of values fit"))

    candidates.sort(key=lambda t: (-t[0], t[2]))
    result = MappingResult()
    used_cols: set[int] = set()
    for score, fname, ci, reason in candidates:
        if score < ACCEPT_THRESHOLD or fname in result.mapping or ci in used_cols:
            continue
        result.mapping[fname] = headers[ci]
        result.confidence[fname] = round(score, 2)
        result.reasons[fname] = reason
        used_cols.add(ci)

    _resolve_address_conflicts(result)
    mapped_cols = set(result.mapping.values())
    result.unmapped_columns = [h for h in headers if h not in mapped_cols]
    _add_warnings(result, headers)
    return result


def _resolve_address_conflicts(result: MappingResult) -> None:
    """A full 'address' column wins over split street parts unless parts are clearly better."""
    m = result.mapping
    if "address" in m and "street_name" in m and "street_number" in m:
        if result.confidence.get("address", 0) < 0.8:
            del m["address"]  # a content-only guess next to explicit street parts is noise
            result.confidence.pop("address", None)
            result.reasons.pop("address", None)
            return
        # Both exist: the split parts usually accompany a redundant display address. Keep both;
        # the importer prefers explicit parts for the street and uses 'address' as a fallback.
        return
    if "address" not in m and "street_name" in m and "street_number" not in m:
        # A lone 'street name' column is really the street line in many exports.
        m["address"] = m.pop("street_name")
        result.confidence["address"] = result.confidence.pop("street_name", 0.6)
        result.reasons["address"] = result.reasons.pop("street_name", "")


def _add_warnings(result: MappingResult, headers: Sequence[str]) -> None:
    if not result.has_address:
        result.warnings.append(
            "No address column found. Pick the column that holds the street address "
            "(or the street number + street name columns) below."
        )
    if "city" not in result.mapping and "address" in result.mapping:
        result.warnings.append(
            "No separate city column: city/state/ZIP will be read from the address text, "
            "falling back to the defaults in Settings."
        )
    for h in headers:
        hn = normalize_header(h)
        if re.search(r"photo|image|picture", hn) and re.search(r"url|link|path", hn):
            result.warnings.append(
                f"Column '{h}' looks like photo URLs. This tool does not download photos from "
                "URLs; use the photo folder import instead (and confirm with your broker what "
                "you may store)."
            )
            break
