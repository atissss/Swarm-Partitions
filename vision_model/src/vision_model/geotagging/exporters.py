"""Export `GeoTaggedDetection` lists to JSON, CSV, and GeoJSON.

These are the pipeline's boundary with downstream consumers (tracking, GIS,
mission planning) — GeoJSON in particular is the standard hand-off format
for QGIS/ArcGIS/PostGIS ingestion (see the architecture doc's extension
points).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vision_model.interfaces.geotag import GeoTaggedDetection
from vision_model.utils.io import ensure_dir

_CSV_FIELDNAMES = [
    "class_name",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pixel_center_x",
    "pixel_center_y",
    "latitude",
    "longitude",
    "altitude",
    "timestamp",
]


def export_json(detections: list[GeoTaggedDetection], path: str | Path, indent: int = 2) -> Path:
    """Write geo-tagged detections to a JSON file (a list of flat dicts).

    Args:
        detections: Detections to export.
        path: Destination `.json` file path.
        indent: Pretty-print indent width.

    Returns:
        The resolved path written to.
    """
    dest = Path(path)
    ensure_dir(dest.parent)
    with dest.open("w", encoding="utf-8") as f:
        json.dump([d.to_dict() for d in detections], f, indent=indent)
    return dest


def export_csv(detections: list[GeoTaggedDetection], path: str | Path) -> Path:
    """Write geo-tagged detections to a flat CSV file.

    Args:
        detections: Detections to export.
        path: Destination `.csv` file path.

    Returns:
        The resolved path written to.
    """
    dest = Path(path)
    ensure_dir(dest.parent)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cx, cy = d.pixel_center
            writer.writerow(
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "pixel_center_x": cx,
                    "pixel_center_y": cy,
                    "latitude": d.latitude,
                    "longitude": d.longitude,
                    "altitude": d.altitude,
                    "timestamp": d.timestamp,
                }
            )
    return dest


def export_geojson(detections: list[GeoTaggedDetection], path: str | Path, indent: int = 2) -> Path:
    """Write geo-tagged detections to a GeoJSON `FeatureCollection`.

    Args:
        detections: Detections to export.
        path: Destination `.geojson` file path.
        indent: Pretty-print indent width.

    Returns:
        The resolved path written to.
    """
    dest = Path(path)
    ensure_dir(dest.parent)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [d.to_geojson_feature() for d in detections],
    }
    with dest.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=indent)
    return dest


def export_all(
    detections: list[GeoTaggedDetection],
    output_dir: str | Path,
    basename: str = "detections",
) -> dict[str, Path]:
    """Write geo-tagged detections in all three formats to `output_dir`.

    Args:
        detections: Detections to export.
        output_dir: Destination directory.
        basename: File stem shared by all three outputs (e.g.
            `"detections"` -> `detections.json`, `detections.csv`,
            `detections.geojson`).

    Returns:
        A dict mapping format name (`"json"`, `"csv"`, `"geojson"`) to the
        resolved path written for that format.
    """
    out_dir = ensure_dir(output_dir)
    return {
        "json": export_json(detections, out_dir / f"{basename}.json"),
        "csv": export_csv(detections, out_dir / f"{basename}.csv"),
        "geojson": export_geojson(detections, out_dir / f"{basename}.geojson"),
    }
