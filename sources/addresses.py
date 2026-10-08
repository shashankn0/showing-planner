"""US address parsing and normalization.

Deliberately dependency-free. Good enough to (a) split common "street, city, ST zip" strings,
(b) build a stable de-duplication key, and (c) format a one-line address for geocoders.
Geocoders do the heavy lifting on messy input; we only need to be consistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

US_STATES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR",
}  # fmt: skip
STATE_CODES = frozenset(US_STATES.values())

STREET_SUFFIXES = {
    "STREET": "ST", "STR": "ST", "AVENUE": "AVE", "AV": "AVE", "BOULEVARD": "BLVD",
    "DRIVE": "DR", "ROAD": "RD", "LANE": "LN", "COURT": "CT", "CIRCLE": "CIR", "PLACE": "PL",
    "TERRACE": "TER", "TRAIL": "TRL", "PARKWAY": "PKWY", "HIGHWAY": "HWY", "SQUARE": "SQ",
    "ALLEY": "ALY", "CROSSING": "XING", "HEIGHTS": "HTS", "POINT": "PT", "RIDGE": "RDG",
    "VALLEY": "VLY", "COVE": "CV", "LOOP": "LOOP", "WAY": "WAY", "PASS": "PASS",
    "CREEK": "CRK", "GROVE": "GRV", "PIKE": "PIKE", "RUN": "RUN", "WALK": "WALK",
}  # fmt: skip
DIRECTIONALS = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "NORTHEAST": "NE",
    "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}  # fmt: skip
UNIT_WORDS = r"(?:APT|APARTMENT|UNIT|STE|SUITE|BLDG|BUILDING|FL|FLOOR|RM|ROOM|LOT|TRLR|#)"
_UNIT_TAIL = re.compile(rf"[\s,]+({UNIT_WORDS})\.?\s*#?\s*([A-Z0-9-]+)\s*$", re.I)
_UNIT_HASH_TAIL = re.compile(r"\s+#\s*([A-Z0-9-]+)\s*$", re.I)
_ZIP = re.compile(r"^\d{5}(?:-\d{4})?$")


@dataclass
class ParsedAddress:
    street: str | None = None
    unit: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None

    @property
    def has_street(self) -> bool:
        return bool(self.street and self.street.strip())


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_state(value: object) -> str | None:
    v = clean(value).upper().rstrip(".")
    if not v:
        return None
    if v in STATE_CODES:
        return v
    return US_STATES.get(v)


def normalize_zip(value: object) -> str | None:
    v = clean(value)
    if not v:
        return None
    digits = re.sub(r"\D", "", v.split("-")[0])
    if len(digits) == 4:  # leading zero eaten by a spreadsheet
        digits = digits.zfill(5)
    return digits if len(digits) == 5 else None


def split_unit(street: str) -> tuple[str, str | None]:
    """'123 Main St Apt 4B' -> ('123 Main St', '4B')."""
    s = clean(street)
    m = _UNIT_TAIL.search(s) or None
    if m:
        return s[: m.start()].strip(" ,"), m.group(2).upper()
    m = _UNIT_HASH_TAIL.search(s)
    if m:
        return s[: m.start()].strip(" ,"), m.group(1).upper()
    return s, None


def normalize_street(street: str) -> str:
    """Uppercase, strip punctuation, abbreviate suffixes and directionals (for keys only)."""
    s = re.sub(r"[^\w\s#-]", " ", clean(street).upper())
    tokens = [t for t in s.split() if t]
    out = []
    for t in tokens:
        out.append(DIRECTIONALS.get(t) or STREET_SUFFIXES.get(t) or t)
    return " ".join(out)


def looks_like_unit(text: str) -> bool:
    return bool(re.match(rf"^\s*{UNIT_WORDS}\.?\s*#?\s*[A-Z0-9-]*\s*$", text, re.I)) or bool(
        re.match(r"^\s*#\s*[A-Z0-9-]+\s*$", text, re.I)
    )


def parse_full_address(
    text: object, default_city: str | None = None, default_state: str | None = None
) -> ParsedAddress:
    """Split '123 Main St, Atlanta, GA 30301' (and common variants) into parts.

    Handles: missing zip/state/city, 'GA, 30301', 'Georgia', unit as its own comma part, and
    'City ST ZIP' with no comma before the state.
    """
    raw = clean(text)
    if not raw:
        return ParsedAddress()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    result = ParsedAddress()

    def take_tail_state_zip(tail: str) -> tuple[str | None, str | None, str | None]:
        """Return (leading_text, state, zip) parsed from a trailing chunk."""
        t = tail.strip()
        m = re.match(r"^(?P<state>[A-Za-z]{2}|[A-Za-z ]{4,})\.?\s*(?P<zip>\d{5}(?:-\d{4})?)?$", t)
        if m and normalize_state(m.group("state")):
            return None, normalize_state(m.group("state")), normalize_zip(m.group("zip"))
        m = re.match(r"^(?P<zip>\d{5}(?:-\d{4})?)$", t)
        if m:
            return None, None, normalize_zip(m.group("zip"))
        m = re.match(r"^(?P<lead>.+?)\s+(?P<state>[A-Za-z]{2})\.?\s*(?P<zip>\d{5}(?:-\d{4})?)?$", t)
        if m and normalize_state(m.group("state")):
            return m.group("lead"), normalize_state(m.group("state")), normalize_zip(m.group("zip"))
        m = re.match(r"^(?P<lead>.+?)\s+(?P<zip>\d{5}(?:-\d{4})?)$", t)
        if m:
            return m.group("lead"), None, normalize_zip(m.group("zip"))
        return t, None, None

    state = zip_ = None
    if len(parts) >= 2:
        lead, state, zip_ = take_tail_state_zip(parts[-1])
        if state or zip_:
            parts = parts[:-1] + ([lead] if lead else [])
            # "GA, 30301" split across two comma parts
            if state and not zip_ and parts and _ZIP.match(parts[-1]):
                zip_ = normalize_zip(parts.pop())
            if zip_ and not state and parts:
                lead2, st2, _ = take_tail_state_zip(parts[-1])
                if st2 and lead2 is None:
                    state = st2
                    parts.pop()
    else:
        # Single chunk: maybe "123 Main St Atlanta GA 30301"
        m = re.match(
            r"^(?P<rest>.+?)\s+(?P<state>[A-Za-z]{2})\.?\s+(?P<zip>\d{5}(?:-\d{4})?)$", raw
        )
        if m and normalize_state(m.group("state")):
            state, zip_ = normalize_state(m.group("state")), normalize_zip(m.group("zip"))
            parts = [m.group("rest")]

    if parts:
        street_part = parts[0]
        rest = parts[1:]
        unit = None
        if rest and looks_like_unit(rest[0]):
            unit = clean(re.sub(rf"^{UNIT_WORDS}\.?\s*#?\s*", "", rest[0], flags=re.I)).upper()
            rest = rest[1:]
        street, unit2 = split_unit(street_part)
        result.street = street
        result.unit = unit or unit2
        if rest:
            result.city = rest[-1].title() if rest[-1].isupper() else rest[-1]
    result.state = state or normalize_state(default_state)
    result.postal_code = zip_
    if not result.city and default_city:
        result.city = clean(default_city)
    return result


def compose_street(
    number: object = None,
    predir: object = None,
    name: object = None,
    suffix: object = None,
    postdir: object = None,
) -> str:
    return clean(" ".join(clean(x) for x in (number, predir, name, suffix, postdir) if clean(x)))


def format_full(
    street: str | None,
    unit: str | None,
    city: str | None,
    state: str | None,
    postal_code: str | None,
) -> str:
    line1 = clean(street)
    if unit:
        line1 = f"{line1} #{unit}" if not str(unit).startswith("#") else f"{line1} {unit}"
    tail = clean(" ".join(x for x in (state or "", postal_code or "") if x))
    return ", ".join(p for p in (line1, clean(city), tail) if p)


def address_key(
    street: str | None,
    unit: str | None,
    city: str | None,
    state: str | None,
    postal_code: str | None,
) -> str:
    """Stable key for matching the same property across exports."""
    parts = [normalize_street(street or ""), (unit or "").upper()]
    if postal_code:
        parts.append(postal_code[:5])
    else:
        parts += [clean(city).upper(), (state or "").upper()]
    return "addr:" + "|".join(parts)


def street_tokens(text: str) -> set[str]:
    """Tokens used to match folder/file names to addresses."""
    return {t for t in normalize_street(text).replace("-", " ").split() if t}
