"""Detection + telemetry -> geo-tagged detection pipeline.

Consumes `interfaces.Detection` (produced by `inference`) and
`interfaces.Telemetry`/`CameraIntrinsics`/`CameraExtrinsics`, and produces
`interfaces.GeoTaggedDetection`. This package never imports from
`inference`, `models`, or `datasets` — only from `interfaces` — so it stays
independently testable and swappable (e.g. a future DTM-aware projection,
Phase 5) without rippling into the detector side of the pipeline.

Public entry points:

    from vision_model.geotagging import tag_detections, tag_single, TelemetryStream
    from vision_model.geotagging.exporters import export_json, export_csv
    from vision_model.geotagging.exporters import export_geojson, export_all
"""

from vision_model.geotagging.geotagger import tag_detections, tag_single
from vision_model.geotagging.telemetry_loader import TelemetryStream, load_telemetry_csv

__all__ = [
    "TelemetryStream",
    "load_telemetry_csv",
    "tag_detections",
    "tag_single",
]
