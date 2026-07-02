"""
kml_loader.py
-------------
Parses a KML file and extracts the mission boundary, no-go zone polygons,
and the optional Home point.

Bug fixes
---------
BUG 7  – buffer(0) is no longer applied blindly. Geometry validity is checked
         explicitly via explain_validity() and repair is logged if needed.
         Previously buffer(0) was called on every polygon unconditionally,
         which could silently alter valid geometry.

BUG 8  – KML parsing is now stricter:
           • "Boundary" matching is case-insensitive.
           • Duplicate Boundary placemarks raise ValueError instead of silently
             overwriting the first.
           • Unnamed placemarks are skipped with a warning rather than being
             added as unnamed no-go zones.
           • A placemark with no <coordinates> element is skipped with a warning.
           • Only <Polygon> placemarks (and recognised <Point> types) are
             processed; other geometry types are warned and skipped.

New features
------------
HOME POINT – A <Point> placemark whose name contains "home" (case-insensitive)
             is now parsed and returned as a (lon, lat) tuple in WGS84.
             Previously all non-polygon placemarks were skipped with a warning.
             Non-polygon, non-home placemarks are still skipped with a warning.
             Duplicate Home placemarks raise ValueError.
             A Home point appearing before any polygon can seed the UTM projection.
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

    BUG 7 FIX: validity is checked explicitly via explain_validity() and any
    repair is logged. buffer(0) is used as the repair mechanism only when needed,
    rather than being applied blindly to every polygon as before.
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


def _init_projection(lon: float, lat: float) -> tuple[str, Transformer, Transformer]:
    """
    Auto-detect the UTM zone from a seed coordinate and return
    (epsg_code, to_meters_transformer, to_latlon_transformer).
    """
    u = utm.from_latlon(lat, lon)
    epsg_code = f"epsg:{'326' if u[3] >= 'N' else '327'}{u[2]:02d}"
    print(f"[kml_loader] Auto-detected UTM zone: {epsg_code}")
    to_meters = Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)
    to_latlon = Transformer.from_crs(epsg_code, "epsg:4326", always_xy=True)
    return epsg_code, to_meters, to_latlon


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_kml(path: str) -> dict:
    """
    Parse a KML file and return boundary, no-go polygons, and optional home point.

    Recognised placemark types
    --------------------------
    Name contains "boundary" (case-insensitive) + has <Polygon>
        → mission boundary

    Name contains "home" (case-insensitive) + has <Point>
        → home coordinate returned as (lon, lat) in WGS84

    Any other named <Polygon> placemark
        → predetermined no-go zone

    Anything else (unnamed, non-polygon, non-home point)
        → skipped with a warning

    Args:
        path: Absolute or relative path to the .kml file.

    Returns:
        A dict with keys:
            "boundary"   – Shapely Polygon in UTM metres
            "nogo"       – list of (name, Shapely Polygon) tuples
            "home"       – (lon, lat) tuple in WGS84, or None if not present
            "epsg_code"  – string e.g. "epsg:32643"
            "to_meters"  – pyproj Transformer (EPSG:4326 → UTM)
            "to_latlon"  – pyproj Transformer (UTM → EPSG:4326)

    Raises:
        ValueError: No Boundary placemark found, duplicate Boundary placemarks,
                    duplicate Home placemarks, or unrepairable polygon geometry.
    """
    tree = etree.parse(path)
    root = tree.getroot()

    to_meters  = None
    to_latlon  = None
    epsg_code  = None
    boundary   = None
    home       = None          # (lon, lat) in WGS84
    nogo: list[tuple[str, Polygon]] = []

    for pm in root.iter(f"{{{NS}}}Placemark"):

        # ── Resolve name ────────────────────────────────────────────────────
        name_el = pm.find(f"{{{NS}}}name")
        name = name_el.text.strip() if (name_el is not None and name_el.text) else ""

        # BUG 8 FIX: unnamed placemarks are skipped with a warning rather than
        # silently becoming unnamed no-go zones.
        if not name:
            print("[kml_loader] WARNING: skipping unnamed placemark.")
            continue

        name_lower = name.lower()

        # ── Detect geometry type ─────────────────────────────────────────────
        has_polygon = pm.find(f".//{{{NS}}}Polygon") is not None
        has_point   = pm.find(f".//{{{NS}}}Point")   is not None

        # ── HOME POINT ───────────────────────────────────────────────────────
        if "home" in name_lower and has_point:
            coords_el = pm.find(f".//{{{NS}}}coordinates")
            if coords_el is None or not coords_el.text:
                print(f"[kml_loader] WARNING: Home placemark '{name}' has no coordinates — skipping.")
                continue

            lon, lat = _parse_coords(coords_el.text)[0]

            if home is not None:
                raise ValueError(
                    f"Duplicate Home placemark found: '{name}'. "
                    "The KML file must contain at most one Home point."
                )

            home = (lon, lat)
            print(f"[kml_loader] Home point loaded: lon={lon:.6f}, lat={lat:.6f}")

            # Use home point to initialise projection if not done yet
            if to_meters is None:
                epsg_code, to_meters, to_latlon = _init_projection(lon, lat)

            continue

        # ── POLYGON PLACEMARKS (boundary + no-go zones) ──────────────────────
        # BUG 8 FIX: non-polygon, non-home placemarks are explicitly skipped
        # with a warning instead of causing silent downstream errors.
        if not has_polygon:
            print(f"[kml_loader] Skipping non-polygon placemark '{name}'.")
            continue

        # BUG 8 FIX: placemarks with missing coordinates are skipped with a
        # warning rather than raising an unhandled exception.
        coords_el = pm.find(f".//{{{NS}}}coordinates")
        if coords_el is None or not coords_el.text:
            print(f"[kml_loader] WARNING: placemark '{name}' has no coordinates — skipping.")
            continue

        coords = _parse_coords(coords_el.text)

        # Initialise projection from first coordinate if not already done
        if to_meters is None:
            epsg_code, to_meters, to_latlon = _init_projection(coords[0][0], coords[0][1])

        m_coords = [to_meters.transform(lon, lat) for lon, lat in coords]
        poly = _make_valid(Polygon(m_coords), label=name)  # BUG 7 FIX: explicit validity check

        # BUG 8 FIX: boundary detection is case-insensitive; duplicate Boundary
        # placemarks raise ValueError instead of silently overwriting the first.
        if "boundary" in name_lower:
            if boundary is not None:
                raise ValueError(
                    f"Duplicate Boundary placemark found: '{name}'. "
                    "The KML file must contain exactly one Boundary polygon."
                )
            boundary = poly
        else:
            nogo.append((name, poly))

    # ── Final validation ─────────────────────────────────────────────────────
    if boundary is None:
        raise ValueError(
            "No 'Boundary' placemark found in the KML file. "
            "Ensure exactly one placemark name contains the word 'boundary' (case-insensitive)."
        )

    if home is None:
        print("[kml_loader] NOTE: No Home point found in KML — 'home' will be null in output.")

    return {
        "boundary":  boundary,
        "nogo":      nogo,
        "home":      home,       # (lon, lat) or None
        "epsg_code": epsg_code,
        "to_meters": to_meters,
        "to_latlon": to_latlon,
    }