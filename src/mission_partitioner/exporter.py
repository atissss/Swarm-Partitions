"""
exporter.py
-----------
Converts in-memory UTM-coordinate polygons back to [longitude, latitude] and
serialises the full mission output to a JSON file.

Bug fixes applied
-----------------
BUG 1  – poly_to_latlon now exports BOTH the exterior ring AND any interior
         rings (holes). Previously only exterior.coords was used, so no-go zones
         that carved holes into partitions were silently dropped from the JSON.

BUG 2  – MultiPolygon is now supported. Each sub-polygon is exported as a
         separate entry in a "rings" list, each with its own exterior and holes.
         Previously calling .exterior on a MultiPolygon raised AttributeError.

BUG 3  – CRS metadata is no longer ambiguous. The JSON now stores both the
         coordinate CRS (always EPSG:4326 – lon/lat) and the planning CRS
         (the UTM zone used internally). Previously the UTM EPSG was written
         as if it described the stored coordinates, which was false.

BUG 4  – random_seed is stored in the JSON so consumers know the generation
         parameters and output is traceable.
"""

import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ring_to_latlon(ring, to_latlon: Transformer) -> list[list[float]]:
    """Convert a single coordinate sequence (exterior or interior ring) to lon/lat."""
    return [list(to_latlon.transform(x, y)) for x, y in ring.coords]


def _polygon_to_geojson_like(poly: Polygon, to_latlon: Transformer) -> dict:
    """
    Serialize one Shapely Polygon to a dict that preserves holes.

    Structure:
        {
            "exterior": [[lon, lat], ...],
            "holes":    [[[lon, lat], ...], ...]   # one list per hole; may be empty
        }
    """
    return {
        "exterior": _ring_to_latlon(poly.exterior, to_latlon),
        "holes": [_ring_to_latlon(ring, to_latlon) for ring in poly.interiors],
    }


def poly_to_latlon(
    poly: Polygon | MultiPolygon | None,
    to_latlon: Transformer,
) -> list[dict]:
    """
    Convert a Shapely Polygon or MultiPolygon (UTM metres) to a list of
    GeoJSON-like ring dicts that fully preserve holes and multi-part geometry.

    Returns [] if poly is None or empty.

    Each element in the returned list represents one simple polygon ring set:
        {
            "exterior": [[lon, lat], ...],
            "holes":    [[[lon, lat], ...], ...]
        }

    A plain Polygon returns a single-element list.
    A MultiPolygon returns one element per constituent polygon.
    """
    if poly is None or poly.is_empty:
        return []

    if isinstance(poly, MultiPolygon):
        return [_polygon_to_geojson_like(p, to_latlon) for p in poly.geoms]

    return [_polygon_to_geojson_like(poly, to_latlon)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_json(
    *,
    boundary: Polygon,
    partitions: list[Polygon | MultiPolygon],
    predetermined_nogo: list[tuple[str, Polygon]],
    dynamic_nogo: list[Polygon],
    to_latlon: Transformer,
    epsg_code: str,
    random_seed: int = 42,
    output_path: str | Path = "data/output/mission_output.json",
) -> Path:
    """
    Serialise all mission data to a JSON file.

    Args:
        boundary:           Mission boundary polygon (UTM metres).
        partitions:         Flyable partition polygons (Polygon or MultiPolygon).
        predetermined_nogo: Static (KML-sourced) no-go zones as (name, poly) tuples.
        dynamic_nogo:       Interactively drawn no-go polygons.
        to_latlon:          pyproj Transformer: UTM → EPSG:4326.
        epsg_code:          Planning CRS string, e.g. "epsg:32643".
        random_seed:        Seed used during sampling/partitioning (stored in metadata).
        output_path:        Destination path; parent directories are created if needed.

    Returns:
        Resolved Path of the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        # -----------------------------------------------------------------
        # BUG 3 FIX: separate coordinate CRS from planning CRS
        # -----------------------------------------------------------------
        "metadata": {
            "crs": {
                "coordinates": "EPSG:4326",
                "axis_order": ["longitude", "latitude"],
                "planning": epsg_code.upper(),
            },
            "n_partitions": len(partitions),
            # BUG 4 FIX: store the seed so output is traceable
            "generation": {
                "random_seed": random_seed,
            },
        },
        # BUG 1+2 FIX: use poly_to_latlon which now handles holes + MultiPolygon
        "boundary": poly_to_latlon(boundary, to_latlon),
        "partitions": [
            {
                "id": i + 1,
                "geometry": poly_to_latlon(p, to_latlon),
            }
            for i, p in enumerate(partitions)
        ],
        "no_go_zones": {
            "predetermined": [
                {
                    "name": name,
                    "geometry": poly_to_latlon(p, to_latlon),
                }
                for name, p in predetermined_nogo
            ],
            "dynamic": [
                {
                    "id": i + 1,
                    "geometry": poly_to_latlon(p, to_latlon),
                }
                for i, p in enumerate(dynamic_nogo)
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"[exporter] JSON written to {output_path.resolve()}")
    return output_path