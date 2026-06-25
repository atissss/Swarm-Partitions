"""
kml_loader.py
-------------
Parses a KML file and extracts the mission boundary and no-go zone polygons.
Automatically detects the appropriate UTM projection from the first coordinate.

Bug fixes applied
-----------------
BUG 7  – buffer(0) is no longer applied blindly. Geometry validity is checked
         explicitly and repair is logged if needed.

BUG 8  – KML parsing is now stricter:
           • "Boundary" matching is case-insensitive.
           • Duplicate Boundary placemarks raise ValueError instead of silently
             overwriting the first.
           • Unnamed placemarks (empty name) are skipped with a warning rather
             than being added as unnamed no-go zones.
           • A placemark with no <coordinates> element is skipped with a warning.
           • Only <Polygon> placemarks are processed; other geometry types
             (Point, LineString, MultiGeometry) are explicitly warned and skipped.
"""

import utm
from lxml import etree
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.validation import explain_validity

NS = "http://www.opengis.net/kml/2.2"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_valid(poly: Polygon, label: str = "") -> Polygon:
    """
    Return a valid version of poly, logging any repair.

    BUG 7 FIX: explicit validity check with logged repair rather than silent
    buffer(0) on every polygon.
    """
    if poly.is_valid:
        return poly

    reason = explain_validity(poly)
    tag = f" [{label}]" if label else ""
    print(f"[kml_loader] WARNING{tag}: invalid geometry — {reason}. Attempting repair.")
    repaired = poly.buffer(0)

    if not repaired.is_valid:
        raise ValueError(
            f"Geometry{tag} could not be repaired: {explain_validity(repaired)}"
        )

    print(f"[kml_loader] Repair succeeded{tag}.")
    return repaired


def _parse_coords(coords_text: str) -> list[tuple[float, float]]:
    """Parse a KML <coordinates> text blob into (lon, lat) tuples."""
    coords = []
    for pt in coords_text.strip().split():
        parts = pt.split(",")
        coords.append((float(parts[0]), float(parts[1])))
    return coords


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_kml(path: str) -> dict:
    """
    Parse a KML file and return boundary + no-go polygons in UTM metre coordinates.

    Args:
        path: Absolute or relative path to the .kml file.

    Returns:
        A dict with keys:
            "boundary"   – Shapely Polygon in UTM metres
            "nogo"       – list of (name, Shapely Polygon) tuples
            "epsg_code"  – string e.g. "epsg:32643"
            "to_meters"  – pyproj Transformer (EPSG:4326 → UTM)
            "to_latlon"  – pyproj Transformer (UTM → EPSG:4326)

    Raises:
        ValueError: No Boundary placemark found, or duplicate Boundary placemarks,
                    or a polygon cannot be repaired.
    """
    tree = etree.parse(path)
    root = tree.getroot()

    to_meters = None
    to_latlon = None
    epsg_code = None
    boundary = None
    nogo: list[tuple[str, Polygon]] = []

    for pm in root.iter(f"{{{NS}}}Placemark"):

        # BUG 8 FIX: skip placemarks that are not polygons
        if pm.find(f".//{{{NS}}}Polygon") is None:
            geom_tags = [child.tag.split("}")[-1] for child in pm]
            print(f"[kml_loader] Skipping non-polygon placemark (tags: {geom_tags}).")
            continue

        # BUG 8 FIX: warn and skip unnamed placemarks
        name_el = pm.find(f"{{{NS}}}name")
        name = name_el.text.strip() if (name_el is not None and name_el.text) else ""
        if not name:
            print("[kml_loader] WARNING: skipping unnamed placemark.")
            continue

        # BUG 8 FIX: warn if no coordinates found
        coords_el = pm.find(f".//{{{NS}}}coordinates")
        if coords_el is None or not coords_el.text:
            print(f"[kml_loader] WARNING: placemark '{name}' has no coordinates — skipping.")
            continue

        coords = _parse_coords(coords_el.text)

        # Initialise projection from the first coordinate encountered
        if to_meters is None:
            u = utm.from_latlon(coords[0][1], coords[0][0])
            epsg_code = f"epsg:{'326' if u[3] >= 'N' else '327'}{u[2]:02d}"
            print(f"[kml_loader] Auto-detected UTM zone: {epsg_code}")
            to_meters = Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)
            to_latlon = Transformer.from_crs(epsg_code, "epsg:4326", always_xy=True)

        m_coords = [to_meters.transform(lon, lat) for lon, lat in coords]
        poly = _make_valid(Polygon(m_coords), label=name)  # BUG 7 FIX

        # BUG 8 FIX: case-insensitive boundary detection + duplicate check
        if "boundary" in name.lower():
            if boundary is not None:
                raise ValueError(
                    f"Duplicate Boundary placemark found: '{name}'. "
                    "The KML file must contain exactly one Boundary polygon."
                )
            boundary = poly
        else:
            nogo.append((name, poly))

    if boundary is None:
        raise ValueError(
            "No 'Boundary' placemark found in the KML file. "
            "Ensure exactly one placemark name contains the word 'boundary' (case-insensitive)."
        )

    return {
        "boundary": boundary,
        "nogo": nogo,
        "epsg_code": epsg_code,
        "to_meters": to_meters,
        "to_latlon": to_latlon,
    }