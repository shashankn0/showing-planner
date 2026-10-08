"""Travel-time estimates.

Phase 3 replaces this with real providers (OpenRouteService / OSRM / Google) behind a cache.
Until then every number from here is a *straight-line estimate* and must be labeled as one.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_MILES = 3958.8
DEFAULT_CIRCUITY = 1.35  # roads are longer than straight lines (a guess; calibrate later)
DEFAULT_MPH = 28.0  # blended metro driving speed (a guess; calibrate later)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = radians(lat1), radians(lat2)
    dphi, dlmb = p2 - p1, radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * asin(sqrt(a))


def estimate_minutes(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    circuity: float = DEFAULT_CIRCUITY,
    mph: float = DEFAULT_MPH,
) -> float:
    """Straight-line distance x circuity / speed, in minutes."""
    return haversine_miles(lat1, lon1, lat2, lon2) * circuity / mph * 60.0
