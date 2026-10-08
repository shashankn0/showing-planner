"""Turn a client's stated preferences + a pool of listings into a transparent tier list.

How it works
  1. Hard requirements (budget cap, minimum beds/baths/size, property type, status, commute
     limit) remove listings, and each removal says why.
  2. Each remaining listing gets 0..1 sub-scores on the criteria the client cares about. A
     sub-score is computed only from facts about the property and the client's own stated needs.
  3. Sub-scores are combined with the client's importance weights into 0..100. Criteria with no
     data are dropped and the rest re-weighted; `coverage` says how much of the weight had data.
  4. Score -> tier (S/A/B/C/D) using fixed cutoffs. Too little data -> "?" instead of a guess.

Nothing here reads or infers neighborhood demographics, and all explanation text is generated
from numbers and the client's own criteria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from statistics import median

from ranking.preferences import CRITERIA, Preferences
from routing.travel import estimate_minutes

MIN_COVERAGE = 0.5
MIN_VALUE_PEERS = 5
TravelFn = Callable[[float, float, float, float], float]


@dataclass
class Candidate:
    """A listing reduced to the facts the ranking may use."""

    id: int
    address: str
    price: int | None = None
    beds: int | None = None
    baths: float | None = None
    sqft: int | None = None
    lot_sqft: int | None = None
    year_built: int | None = None
    property_type: str | None = None
    status: str | None = None
    lat: float | None = None
    lon: float | None = None
    list_date: date | None = None
    condition: float | None = None  # 1..5 from photo scoring, when available
    condition_photos: int = 0
    mls: str | None = None
    is_synthetic: bool = False


@dataclass
class Factor:
    key: str
    label: str
    score: float | None  # 0..1, None = no data
    weight: float
    detail: str


@dataclass
class Ranked:
    candidate: Candidate
    score: float | None = None
    tier: str = "?"  # S A B C D, "?" = not enough data, "X" = excluded
    coverage: float = 0.0
    factors: list[Factor] = field(default_factory=list)
    excluded_reasons: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    watchouts: list[str] = field(default_factory=list)


@dataclass
class RankingResult:
    ranked: list[Ranked]
    excluded: list[Ranked]

    def by_tier(self) -> dict[str, list[Ranked]]:
        out: dict[str, list[Ranked]] = {}
        for r in self.ranked:
            out.setdefault(r.tier, []).append(r)
        return out


def _n(n: float, word: str) -> str:
    """'1 bedroom', '2 bedrooms', '2.5 bathrooms'."""
    return f"{n:g} {word}" if n == 1 else f"{n:g} {word}s"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ hard requirements


def _exclusions(
    c: Candidate, p: Preferences, commutes: dict[str, float | None]
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    unknowns: list[str] = []

    def need(value, label: str):  # noqa: ANN001, ANN202
        if value is None:
            unknowns.append(f"{label} not listed")
            return False
        return True

    if p.max_price is not None and need(c.price, "Price") and c.price > p.max_price:
        reasons.append(f"Price ${c.price:,} is over the ${p.max_price:,} maximum")
    if p.min_price is not None and need(c.price, "Price") and c.price < p.min_price:
        reasons.append(f"Price ${c.price:,} is under the ${p.min_price:,} minimum")
    if p.min_beds is not None and need(c.beds, "Bedrooms") and c.beds < p.min_beds:
        reasons.append(f"{_n(c.beds, 'bedroom')}; needs {p.min_beds}+")
    if p.min_baths is not None and need(c.baths, "Bathrooms") and c.baths < p.min_baths:
        reasons.append(f"{_n(c.baths, 'bathroom')}; needs {p.min_baths:g}+")
    if p.min_sqft is not None and need(c.sqft, "Square footage") and c.sqft < p.min_sqft:
        reasons.append(f"{c.sqft:,} sq ft; needs {p.min_sqft:,}+")
    if (
        p.min_year_built is not None
        and need(c.year_built, "Year built")
        and c.year_built < p.min_year_built
    ):
        reasons.append(f"Built {c.year_built}; needs {p.min_year_built}+")
    if p.property_types:
        if c.property_type is None:
            unknowns.append("Property type not listed")
        else:
            t = c.property_type.lower()
            if not any(w.lower() in t or t in w.lower() for w in p.property_types):
                reasons.append(f"Property type is {c.property_type}")
    if p.statuses and c.status is not None and c.status not in p.statuses:
        reasons.append(f"Status is {c.status.replace('_', ' ')}")
    for d in p.destinations:
        m = commutes.get(d.name)
        if d.max_minutes is not None and m is not None and m > d.max_minutes:
            reasons.append(f"≈{m:.0f} min to {d.name}; limit is {d.max_minutes:.0f}")
    return reasons, unknowns


# ------------------------------------------------------------------ criteria


def _price_fit(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    target = p.target_price or (int(0.9 * p.max_price) if p.max_price else None)
    if c.price is None or target is None:
        return None, ""
    if c.price <= target:
        return 1.0, f"${c.price:,}, at or under the ${target:,} target"
    cap = p.max_price or int(target * 1.15)
    span = max(cap - target, 1)
    score = 1.0 - 0.6 * _clamp((c.price - target) / span)
    return score, f"${c.price:,}, ${c.price - target:,} over the ${target:,} target"


def _beds_baths(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    parts, bits = [], []
    if p.min_beds is not None and c.beds is not None:
        parts.append(min(1.0, 0.7 + 0.15 * max(c.beds - p.min_beds, 0)))
        bits.append(f"{c.beds} bd (wanted {p.min_beds}+)")
    if p.min_baths is not None and c.baths is not None:
        parts.append(min(1.0, 0.7 + 0.15 * max((c.baths - p.min_baths) * 2, 0) / 2))
        bits.append(f"{c.baths:g} ba (wanted {p.min_baths:g}+)")
    if not parts:
        return None, ""
    return sum(parts) / len(parts), ", ".join(bits)


def _space(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    if c.sqft is None or (p.min_sqft is None and p.ideal_sqft is None):
        return None, ""
    lo = p.min_sqft or 0
    ideal = p.ideal_sqft or (int(lo * 1.25) if lo else None)
    if ideal is None:
        return None, ""
    if c.sqft >= ideal:
        score = 1.0
    elif ideal > lo:
        score = 0.5 + 0.5 * _clamp((c.sqft - lo) / (ideal - lo))
    else:
        score = 1.0
    return score, f"{c.sqft:,} sq ft (ideal {ideal:,})"


def _commute(
    c: Candidate, p: Preferences, commutes: dict[str, float | None]
) -> tuple[float | None, str]:
    scored, bits = [], []
    for d in p.destinations:
        m = commutes.get(d.name)
        if m is None:
            continue
        ref = d.max_minutes or 40.0
        scored.append((_clamp((ref - m) / (0.75 * ref)), max(d.weight, 0.0)))
        bits.append(f"≈{m:.0f} min to {d.name}")
    total_w = sum(w for _, w in scored)
    if not scored or total_w == 0:
        return None, ""
    score = sum(s * w for s, w in scored) / total_w
    return score, "; ".join(bits) + " (straight-line estimate)"


def _condition(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    if c.condition is None or p.condition_mode == "neutral":
        return None, ""
    good = _clamp((c.condition - 1) / 4)
    # renovation_ok: a project is welcome, but severe damage is still a risk
    score = good if p.condition_mode == "move_in_ready" else 0.3 + 0.7 * (1 - good)
    return score, f"{c.condition:.1f}/5 from {c.condition_photos} photo(s)"


def _age(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    if c.year_built is None or p.condition_mode == "renovation_ok":
        return None, ""
    return _clamp((c.year_built - 1950) / 75), f"Built {c.year_built}"


def _lot(c: Candidate, p: Preferences) -> tuple[float | None, str]:
    if c.lot_sqft is None or p.ideal_lot_sqft is None:
        return None, ""
    return _clamp(c.lot_sqft / p.ideal_lot_sqft), f"{c.lot_sqft:,} sq ft lot"


def _negotiation(c: Candidate, today: date) -> tuple[float | None, str]:
    if c.list_date is None:
        return None, ""
    dom = max((today - c.list_date).days, 0)
    return _clamp(dom / 90), f"{dom} days on market"


def _value_stats(pool: list[Candidate]) -> dict[str, tuple[float, int]]:
    by_type: dict[str, list[float]] = {}
    for c in pool:
        if c.price and c.sqft:
            by_type.setdefault((c.property_type or "").lower(), []).append(c.price / c.sqft)
    return {t: (median(v), len(v)) for t, v in by_type.items() if len(v) >= MIN_VALUE_PEERS}


def _value(c: Candidate, stats: dict[str, tuple[float, int]]) -> tuple[float | None, str]:
    if not (c.price and c.sqft):
        return None, ""
    st = stats.get((c.property_type or "").lower())
    if st is None:
        return None, ""
    med, n = st
    ppsf = c.price / c.sqft
    return _clamp(
        (1.4 - ppsf / med) / 0.6
    ), f"${ppsf:,.0f}/sq ft vs. ${med:,.0f} median of {n} homes"


# ------------------------------------------------------------------ main entry


def rank(
    candidates: list[Candidate],
    prefs: Preferences,
    today: date | None = None,
    travel: TravelFn = estimate_minutes,
) -> RankingResult:
    prefs.validate()
    today = today or date.today()
    weights = prefs.effective_weights()

    commutes: dict[int, dict[str, float | None]] = {}
    for c in candidates:
        commutes[c.id] = {
            d.name: (
                travel(c.lat, c.lon, d.lat, d.lon)
                if c.lat is not None and c.lon is not None
                else None
            )
            for d in prefs.destinations
        }

    kept: list[Ranked] = []
    excluded: list[Ranked] = []
    for c in candidates:
        reasons, unknowns = _exclusions(c, prefs, commutes[c.id])
        r = Ranked(candidate=c, excluded_reasons=reasons, unknowns=unknowns)
        (excluded if reasons else kept).append(r)
    for r in excluded:
        r.tier = "X"

    stats = _value_stats([r.candidate for r in kept])
    applicable = {
        "price_fit": bool(prefs.max_price or prefs.target_price),
        "beds_baths": prefs.min_beds is not None or prefs.min_baths is not None,
        "space": prefs.min_sqft is not None or prefs.ideal_sqft is not None,
        "value": bool(stats),
        "commute": bool(prefs.destinations),
        "condition": prefs.condition_mode != "neutral",
        "age": prefs.condition_mode != "renovation_ok",
        "lot": prefs.ideal_lot_sqft is not None,
        "negotiation": True,
    }

    for r in kept:
        c = r.candidate
        raw = {
            "price_fit": _price_fit(c, prefs),
            "beds_baths": _beds_baths(c, prefs),
            "space": _space(c, prefs),
            "value": _value(c, stats),
            "commute": _commute(c, prefs, commutes[c.id]),
            "condition": _condition(c, prefs),
            "age": _age(c, prefs),
            "lot": _lot(c, prefs),
            "negotiation": _negotiation(c, today),
        }
        r.factors = [
            Factor(k, CRITERIA[k], raw[k][0], weights[k], raw[k][1])
            for k in CRITERIA
            if applicable[k] and weights[k] > 0
        ]
        total_w = sum(f.weight for f in r.factors)
        have = [f for f in r.factors if f.score is not None]
        have_w = sum(f.weight for f in have)
        r.coverage = have_w / total_w if total_w else 0.0
        if have_w > 0:
            r.score = round(100 * sum(f.weight * f.score for f in have) / have_w, 1)
        r.tier = _tier(r, prefs)
        _explain(r)

    kept.sort(key=lambda r: (-(r.score or -1), -r.coverage, r.candidate.price or 10**12))
    excluded.sort(key=lambda r: r.candidate.address)
    return RankingResult(ranked=kept, excluded=excluded)


def _tier(r: Ranked, p: Preferences) -> str:
    if r.score is None or r.coverage < MIN_COVERAGE:
        return "?"
    for t in ("S", "A", "B", "C"):
        if r.score >= p.tier_cutoffs[t]:
            return t
    return "D"


def _explain(r: Ranked) -> None:
    scored = [f for f in r.factors if f.score is not None and f.detail]
    strong = sorted((f for f in scored if f.score >= 0.8), key=lambda f: -f.weight * f.score)
    weak = sorted((f for f in scored if f.score <= 0.45), key=lambda f: -f.weight * (1 - f.score))
    r.strengths = [f"{f.label}: {f.detail}" for f in strong[:3]]
    r.watchouts = [f"{f.label}: {f.detail}" for f in weak[:3]]
    missing = [f.label for f in r.factors if f.score is None]
    if missing:
        r.unknowns.append("No data for: " + ", ".join(missing))
