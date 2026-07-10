"""Stable final output contract, consumed by downstream modules
(tracking, GIS, mission planning, analytics).

This shape must remain stable across pipeline changes; bump a schema
version rather than breaking field names/types if it ever needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from vision_model.interfaces.detection import ClassName


@dataclass(frozen=True, slots=True)
class GeoTaggedDetection:
    """A detection enriched with an estimated geographic location.

    Attributes:
        class_name: Detected object category.
        confidence: Model confidence score in [0.0, 1.0].
        bbox: Bounding box (x1, y1, x2, y2) in source-image pixel
            coordinates.
        pixel_center: (cx, cy) pixel coordinates used as the projection
            anchor.
        latitude: Estimated latitude in decimal degrees (WGS84).
        longitude: Estimated longitude in decimal degrees (WGS84).
        altitude: Estimated ground elevation at the detection, in meters
            (mean sea level, or above the reference DTM if one was used).
        timestamp: ISO-8601 UTC timestamp of the source frame.
    """

    class_name: ClassName
    confidence: float
    bbox: tuple[float, float, float, float]
    pixel_center: tuple[float, float]
    latitude: float
    longitude: float
    altitude: float
    timestamp: str

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")

    def to_dict(self) -> dict:
        """Serialize to a flat, JSON-friendly dict."""
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "pixel_center": list(self.pixel_center),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "timestamp": self.timestamp,
        }

    def to_geojson_feature(self) -> dict:
        """Serialize to a GeoJSON Point Feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.longitude, self.latitude, self.altitude],
            },
            "properties": {
                "class_name": self.class_name,
                "confidence": self.confidence,
                "bbox": list(self.bbox),
                "pixel_center": list(self.pixel_center),
                "timestamp": self.timestamp,
            },
        }
