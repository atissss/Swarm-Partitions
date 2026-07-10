"""Orchestrates `Detection` + `Telemetry` + camera model -> `GeoTaggedDetection`.

This is the single public entry point the rest of the pipeline (and
downstream consumers) should call — `inference` and `datasets` never import
anything from this module; `geotagger` imports `interfaces.Detection` only,
never `inference` or `models` internals.
"""

from __future__ import annotations

from vision_model.geotagging.camera_model import pixel_to_ray
from vision_model.geotagging.frames import camera_ray_to_ned, ned_offset_to_geodetic
from vision_model.geotagging.projection import intersect_flat_ground
from vision_model.geotagging.telemetry_loader import TelemetryStream
from vision_model.interfaces.camera import CameraExtrinsics, CameraIntrinsics
from vision_model.interfaces.detection import Detection
from vision_model.interfaces.geotag import GeoTaggedDetection
from vision_model.interfaces.telemetry import Telemetry
from vision_model.utils.exceptions import GeoTagError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

TelemetrySource = Telemetry | TelemetryStream | list[Telemetry]


def _resolve_telemetry(
    detection: Detection,
    telemetry_source: TelemetrySource,
    max_telemetry_delta_seconds: float | None,
) -> Telemetry:
    """Resolve the `Telemetry` sample to use for one detection.

    A single `Telemetry` is used as-is (e.g. a hovering shot or a
    single-frame capture where time-sync is moot). A `TelemetryStream` or
    raw `list[Telemetry]` is searched for the nearest sample to the
    detection's timestamp.
    """
    if isinstance(telemetry_source, Telemetry):
        return telemetry_source

    stream = (
        telemetry_source
        if isinstance(telemetry_source, TelemetryStream)
        else TelemetryStream(telemetry_source)
    )
    return stream.nearest(detection.timestamp, max_delta_seconds=max_telemetry_delta_seconds)


def tag_single(
    detection: Detection,
    telemetry_source: TelemetrySource,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics | None = None,
    ground_elevation_msl: float = 0.0,
    max_telemetry_delta_seconds: float | None = None,
) -> GeoTaggedDetection:
    """Geo-tag a single `Detection`.

    Args:
        detection: The detection to geo-tag (pixel-space only).
        telemetry_source: A single `Telemetry` (used as-is), or a
            `TelemetryStream`/`list[Telemetry]` (nearest sample to
            `detection.timestamp` is used).
        intrinsics: Camera intrinsics for pixel -> ray back-projection.
        extrinsics: Fixed camera mount offset. Defaults to a rigid,
            co-located mount (no offset) when omitted.
        ground_elevation_msl: Assumed flat-ground elevation in meters MSL
            (the flat-earth model — see `projection.intersect_flat_ground`
            and the Phase 5 DTM-aware extension point).
        max_telemetry_delta_seconds: If `telemetry_source` is a stream/list,
            raise if the nearest sample is farther than this from
            `detection.timestamp`.

    Returns:
        A `GeoTaggedDetection` with `latitude`/`longitude`/`altitude`
        populated.

    Raises:
        GeoTagError: If telemetry can't be resolved, or the ray doesn't
            meaningfully intersect the ground plane (see
            `projection.intersect_flat_ground`).
    """
    extrinsics = extrinsics if extrinsics is not None else CameraExtrinsics()
    telemetry = _resolve_telemetry(detection, telemetry_source, max_telemetry_delta_seconds)

    height_above_ground = telemetry.altitude_msl - ground_elevation_msl

    ray_camera = pixel_to_ray(detection.pixel_center, intrinsics)
    ray_ned = camera_ray_to_ned(ray_camera, telemetry, extrinsics)
    ned_offset = intersect_flat_ground(ray_ned, height_above_ground)
    latitude, longitude, altitude = ned_offset_to_geodetic(
        ned_offset, telemetry.latitude, telemetry.longitude, telemetry.altitude_msl
    )

    return GeoTaggedDetection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=detection.bbox,
        pixel_center=detection.pixel_center,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        timestamp=detection.timestamp,
    )


def tag_detections(
    detections: list[Detection],
    telemetry_source: TelemetrySource,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics | None = None,
    ground_elevation_msl: float = 0.0,
    max_telemetry_delta_seconds: float | None = None,
    skip_errors: bool = False,
) -> list[GeoTaggedDetection]:
    """Geo-tag a batch of detections.

    Args:
        detections: Detections to geo-tag.
        telemetry_source: See `tag_single`.
        intrinsics: Camera intrinsics.
        extrinsics: Fixed camera mount offset. Defaults to a rigid,
            co-located mount (no offset) when omitted.
        ground_elevation_msl: Assumed flat-ground elevation in meters MSL.
        max_telemetry_delta_seconds: See `tag_single`.
        skip_errors: If False (default), a `GeoTagError` on any detection
            propagates immediately — "fail loud" rather than silently
            dropping a bad geo-tag. If True, the failing detection is
            logged and skipped so the rest of the batch still completes;
            use this for large batch/video runs where a handful of
            horizon-grazing detections shouldn't abort the whole export.

    Returns:
        `GeoTaggedDetection`s for every input detection that geo-tagged
        successfully, same relative order.

    Raises:
        GeoTagError: If `skip_errors` is False and any detection fails to
            geo-tag.
    """
    extrinsics = extrinsics if extrinsics is not None else CameraExtrinsics()

    # Resolve telemetry into a reusable stream once, rather than per-detection,
    # when given a raw list (avoids re-sorting on every call).
    resolved_source: TelemetrySource = (
        TelemetryStream(telemetry_source)
        if isinstance(telemetry_source, list)
        else telemetry_source
    )

    results: list[GeoTaggedDetection] = []
    for detection in detections:
        try:
            results.append(
                tag_single(
                    detection,
                    resolved_source,
                    intrinsics,
                    extrinsics,
                    ground_elevation_msl,
                    max_telemetry_delta_seconds,
                )
            )
        except GeoTagError as exc:
            if not skip_errors:
                raise
            logger.warning(
                "Skipping detection (frame_id=%s, class=%s): %s",
                detection.frame_id,
                detection.class_name,
                exc,
            )

    return results
