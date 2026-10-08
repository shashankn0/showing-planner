"""Demo listings for trying the app without any data. Entirely synthetic.

Every address uses made-up street and town names, MLS numbers start with DEMO, coordinates are
random points around metro Atlanta (they do not match the addresses), and each listing is flagged
`is_synthetic` so the UI labels it as demo data.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from sources.base import RawListing

STREETS = [
    "Placeholder", "Example", "Fictional", "Sample", "Synthetic",
    "Mockup", "Pretend", "Testing", "Imaginary", "Demo",
]  # fmt: skip
SUFFIXES = ["St", "Ave", "Dr", "Ln", "Ct", "Way", "Pl", "Rd", "Cir", "Trl"]
TOWNS = [
    ("Demoville", "30301"), ("Testburg", "30302"), ("Samplewood", "30303"),
    ("Fakeland", "30304"), ("Mockton", "30305"), ("Pretendia", "30306"),
]  # fmt: skip
CENTER = (33.75, -84.39)  # metro Atlanta, for plausible distances
ACCESS = ["go_direct", "call_first", "appointment", "appointment"]


DEFAULT_COUNT = 60


def demo_listings(
    n: int = DEFAULT_COUNT, seed: int = 7, today: date | None = None
) -> list[RawListing]:
    """A plausible mix: mostly 3-4 bedroom homes, prices tied to size, all within ~9 miles."""
    rng = random.Random(seed)
    today = today or date.today()
    out: list[RawListing] = []
    for i in range(n):
        kind = rng.choices(["Single Family Residential", "Townhouse", "Condo"], [6, 2, 2])[0]
        if kind == "Condo":
            beds = rng.choices([1, 2, 3], [3, 5, 2])[0]
        else:
            beds = rng.choices([2, 3, 4, 5], [2, 5, 4, 1])[0]
        scale = 0.8 if kind != "Single Family Residential" else 1.0
        sqft = int(round(scale * (500 + beds * 380) * rng.uniform(0.8, 1.2), -1))
        baths = max(1.0, min(4.5, beds - rng.choice([0, 0, 1]) + rng.choice([0.0, 0.5])))
        ppsf = min(380.0, max(150.0, rng.gauss(240, 50)))
        price = int(round(sqft * ppsf, -3))
        full = int(baths)
        half = 1 if baths - full >= 0.5 else 0
        status = rng.choices(["active", "coming_soon", "pending"], [8, 1, 1])[0]
        town, zip_code = rng.choice(TOWNS)
        number = rng.randint(100, 9800)
        unit = str(rng.randint(1, 40)) if kind == "Condo" else None
        out.append(
            RawListing(
                source="demo",
                street=f"{number} {rng.choice(STREETS)} {rng.choice(SUFFIXES)}",
                unit=unit,
                city=town,
                state="GA",
                postal_code=zip_code,
                source_listing_id=f"DEMO{1000 + i}",
                latitude=round(CENTER[0] + rng.uniform(-0.13, 0.13), 6),
                longitude=round(CENTER[1] + rng.uniform(-0.16, 0.16), 6),
                list_price=price,
                status=status,
                property_type=kind,
                beds=beds,
                baths_full=full,
                baths_half=half,
                baths_total=baths,
                sqft=sqft,
                lot_sqft=rng.randint(4000, 22000) if kind == "Single Family Residential" else None,
                year_built=rng.randint(1955, 2024),
                list_date=today - timedelta(days=rng.randint(1, 120)),
                access_type=rng.choice(ACCESS),
                is_synthetic=True,
            )
        )
    return out
