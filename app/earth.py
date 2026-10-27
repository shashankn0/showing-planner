"""Google viewing helpers: links, a KML file for Google Earth, and an embedded 3D map.

No Streamlit here, so it is testable. Google Earth itself has no documented embed or URL scheme,
only file import, so Earth gets a KML download; the in-page 3D view uses the Maps JavaScript API.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from app.services import SavedItem

_XML_BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # not allowed in XML 1.0


def maps_url(address: str) -> str:
    """Google Maps search for an address. Needs no API key."""
    return "https://www.google.com/maps/search/?" + urlencode({"api": 1, "query": address})


def satellite_url(lat: float, lon: float) -> str:
    """Google Maps satellite view centered on a point. Needs no API key."""
    return "https://www.google.com/maps/@?" + urlencode(
        {
            "api": 1,
            "map_action": "map",
            "center": f"{lat},{lon}",
            "zoom": 19,
            "basemap": "satellite",
        }
    )


def _xml(text: str) -> str:
    return escape(_XML_BAD.sub("", text))


def kml(homes: list[SavedItem]) -> str:
    """Saved homes as KML. Google Earth imports it from File > Import. Notes are left out."""
    marks = "".join(
        f"<Placemark><name>{_xml(h.address)}</name>"
        f"<description>{_xml(h.status + (f' · ${h.price:,}' if h.price else ''))}</description>"
        f"<Point><coordinates>{h.lon:.6f},{h.lat:.6f},0</coordinates></Point></Placemark>"
        for h in homes
        if h.lat is not None and h.lon is not None
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Saved homes</name>'
        f"{marks}</Document></kml>"
    )


def viewer_html(lat: float, lon: float, api_key: str, height: int = 520) -> str:
    """A photorealistic 3D map centered on a home (Maps JavaScript API, `maps3d` library).

    The map's center altitude is meters above sea level, so the ground height is looked up
    first from USGS (US only; keyless). If that fails the map still opens at altitude 0.
    Each map load is billed by Google; the page only renders this on request.
    """
    lat, lon = float(lat), float(lon)
    key = quote(api_key, safe="")
    return f"""<style>body{{margin:0}}gmp-map-3d{{display:block;height:{height}px;width:100%}}</style>
<script async src="https://maps.googleapis.com/maps/api/js?loading=async&key={key}&libraries=maps3d"></script>
<gmp-map-3d id="m" mode="hybrid" center="{lat:.6f},{lon:.6f},0" range="450" tilt="65" heading="0"></gmp-map-3d>
<script>
fetch("https://epqs.nationalmap.gov/v1/json?x={lon:.6f}&y={lat:.6f}&units=Meters&wkid=4326&includeDate=false")
  .then(r => r.json())
  .then(j => {{
    const a = parseFloat(j.value);
    if (a > -500) document.getElementById("m").setAttribute("center", "{lat:.6f},{lon:.6f}," + a);
  }})
  .catch(() => {{}});
</script>"""
