"""Tolerant parsers for the messy values found in MLS / portal exports."""

from __future__ import annotations

import re
from datetime import date, datetime

_NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")
_MISSING = {"", "n/a", "na", "none", "null", "nan", "-", "--", "unknown", "tbd"}


def is_missing(value: object) -> bool:
    return str(value if value is not None else "").strip().lower() in _MISSING


def to_float(value: object) -> float | None:
    if is_missing(value):
        return None
    m = _NUM.search(str(value).replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def to_int(value: object) -> int | None:
    f = to_float(value)
    return None if f is None else int(round(f))


def parse_price(value: object) -> int | None:
    """'$425,000', '425000.00', '425K', '1.2M' -> int dollars."""
    if is_missing(value):
        return None
    s = str(value).strip().lower().replace("$", "").replace(",", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([km])?\b", s)
    if not m:
        return None
    n = float(m.group(1))
    if m.group(2) == "k":
        n *= 1_000
    elif m.group(2) == "m":
        n *= 1_000_000
    return int(round(n))


def parse_baths(value: object) -> tuple[int | None, int | None, float | None]:
    """Return (full, half, total) from '2.5', '2/1', '2 full 1 half', '2F 1H', '3'."""
    if is_missing(value):
        return None, None, None
    s = str(value).strip().lower()
    m = re.search(r"(\d+)\s*(?:f|full)\b[^\d]*(\d+)\s*(?:h|half|p|partial)\b", s)
    if m:
        full, half = int(m.group(1)), int(m.group(2))
        return full, half, full + 0.5 * half
    m = re.match(r"^(\d+)\s*[/|:]\s*(\d+)$", s)
    if m:
        full, half = int(m.group(1)), int(m.group(2))
        return full, half, full + 0.5 * half
    f = to_float(s)
    if f is None:
        return None, None, None
    full = int(f)
    half = 1 if (f - full) >= 0.5 else 0
    return full, half, f


def parse_date(value: object) -> date | None:
    if is_missing(value):
        return None
    s = str(value).strip()
    for fmt in (
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    ):  # fmt: skip
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


_STATUS_RULES = (
    ("coming_soon", ("coming soon",)),
    ("pending", ("pending", "under contract", "contingent", "ucb", "option", "active under")),
    ("closed", ("closed", "sold", "settled")),
    ("off_market", ("withdrawn", "cancel", "expired", "off market", "terminated")),
    ("hold", ("hold",)),
    ("active", ("active", "for sale", "new", "price change", "back on market", "bom")),
)


def normalize_status(value: object) -> str | None:
    if is_missing(value):
        return None
    s = str(value).strip().lower()
    for canonical, needles in _STATUS_RULES:
        if any(n in s for n in needles):
            return canonical
    if s in {"a", "act"}:
        return "active"
    if s in {"p", "pend"}:
        return "pending"
    if s in {"c", "s"}:
        return "closed"
    return s[:32]


ACCESS_TYPES = ("go_direct", "call_first", "appointment", "unknown")


def classify_access(text: object) -> str:
    """Best-effort suggestion from free-text showing instructions. Editable in the UI.

    Heuristic only: the listing's own instructions always win, and the agent must follow them.
    """
    if is_missing(text):
        return "unknown"
    s = str(text).lower()
    if re.search(r"appointment|appt|showingtime|showing time|schedule", s):
        return "appointment"
    if re.search(r"call|text|notice|occupied|tenant|owner|\b(24|48) ?h(ou)?r", s):
        return "call_first"
    if re.search(r"lock ?box|supra|sentri|go direct|go-direct|key ?any|vacant|combo", s):
        return "go_direct"
    return "unknown"


_CODE_WORDS = r"(?:code|alarm|combo|combination|pin|passcode|password|gate|lockbox|lock box)"
_CODE_RE = re.compile(rf"(?i)\b({_CODE_WORDS})\b([^\n.;]{{0,24}}?)(#?\d{{3,8}}#?)")


def redact_access_codes(text: object) -> str | None:
    """Mask digits that follow words like 'code' / 'alarm' / 'lockbox' so we never store them.

    Not a security control; a guardrail against accidentally keeping door/alarm codes in a
    local database when showing-instruction fields are imported.
    """
    if is_missing(text):
        return None
    return _CODE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}****", str(text).strip())
