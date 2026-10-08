"""What a client is looking for, in their own words.

Fair-housing design rule: every ranking criterion is listed here, and `Preferences.from_dict`
rejects anything else. Criteria describe the *property* (price, size, condition, age, lot) or
the *client's own stated needs* (commute to places they name, budget). There is deliberately no
way to express neighborhood demographics, and no free-form "neighborhood quality" score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# key -> label shown in the UI. This is the complete allowlist of ranking criteria.
CRITERIA: dict[str, str] = {
    "price_fit": "Price vs. budget",
    "beds_baths": "Bedrooms and bathrooms",
    "space": "Living space",
    "value": "Price per sq ft vs. the other homes being compared",
    "commute": "Estimated drive time to the places they named",
    "condition": "Condition from photos (once scored)",
    "age": "Year built",
    "lot": "Lot size",
    "negotiation": "Time on market (room to negotiate)",
}
DEFAULT_WEIGHTS: dict[str, float] = {
    "price_fit": 8,
    "beds_baths": 7,
    "space": 5,
    "value": 5,
    "commute": 6,
    "condition": 7,
    "age": 2,
    "lot": 2,
    "negotiation": 0,
}
CONDITION_MODES = {
    "neutral": "Don't factor in condition",
    "move_in_ready": "Wants move-in ready",
    "renovation_ok": "Open to a project / fixer-upper",
}
TIERS = ("S", "A", "B", "C", "D")
DEFAULT_CUTOFFS = {"S": 85.0, "A": 72.0, "B": 58.0, "C": 45.0}  # below C -> D
DEFAULT_STATUSES = ["active", "coming_soon"]


@dataclass
class Destination:
    """A place the client named (work, family, school drop-off). Never inferred."""

    name: str
    lat: float
    lon: float
    weight: float = 1.0
    max_minutes: float | None = None  # hard limit if set
    address: str = ""  # as typed, kept so a saved profile can show it again


@dataclass
class Preferences:
    # budget
    max_price: int | None = None
    min_price: int | None = None
    target_price: int | None = None
    # the home
    min_beds: int | None = None
    min_baths: float | None = None
    min_sqft: int | None = None
    ideal_sqft: int | None = None
    ideal_lot_sqft: int | None = None
    min_year_built: int | None = None
    property_types: list[str] = field(default_factory=list)  # empty = any
    statuses: list[str] = field(default_factory=lambda: list(DEFAULT_STATUSES))
    # the client's places
    destinations: list[Destination] = field(default_factory=list)
    # how the client feels about condition (chosen by the agent from what the client said)
    condition_mode: str = "neutral"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    tier_cutoffs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CUTOFFS))

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        unknown = set(self.weights) - set(CRITERIA)
        if unknown:
            raise ValueError(
                f"Unknown ranking criteria {sorted(unknown)}. Allowed: {sorted(CRITERIA)}."
            )
        if any(w < 0 or w > 10 for w in self.weights.values()):
            raise ValueError("Importance weights must be between 0 and 10.")
        if self.condition_mode not in CONDITION_MODES:
            raise ValueError(f"condition_mode must be one of {sorted(CONDITION_MODES)}.")
        if (
            self.max_price is not None
            and self.min_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("Minimum price is above maximum price.")
        cut = [self.tier_cutoffs.get(t) for t in ("S", "A", "B", "C")]
        if None in cut or cut != sorted(cut, reverse=True):
            raise ValueError("Tier cutoffs must be S > A > B > C.")

    def effective_weights(self) -> dict[str, float]:
        w = {k: float(self.weights.get(k, DEFAULT_WEIGHTS[k])) for k in CRITERIA}
        if self.condition_mode == "neutral":
            w["condition"] = 0.0
        return w

    # ------------------------------------------------------------------ (de)serialization

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Preferences:
        """Strict: unknown keys raise instead of being silently accepted."""
        raw = dict(raw)
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown preference fields: {sorted(unknown)}.")
        dests = [Destination(**d) for d in raw.pop("destinations", [])]
        prefs = cls(destinations=dests, **raw)
        prefs.validate()
        return prefs
