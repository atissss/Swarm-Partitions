from __future__ import annotations

import pytest

from vision_model.geotagging.geotagger import tag_detections, tag_single
from vision_model.geotagging.telemetry_loader import TelemetryStream
from vision_model.interfaces.detection import Detection
from vision_model.interfaces.telemetry import Telemetry
from vision_model.utils.exceptions import GeoTagError, TelemetryNotFoundError


def _det(pixel_center, timestamp="2026-07-01T15:42:11Z", bbox=None) -> Detection:
    cx, cy = pixel_center
    bbox = bbox or (cx - 10, cy - 10, cx + 10, cy + 10)
    return Detection(
        class_name="car",
        confidence=0.95,
        bbox=bbox,
        pixel_center=pixel_center,
        frame_id="f",
        timestamp=timestamp,
    )


class TestTagSingle:
    def test_nadir_center_pixel_matches_drone_position(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        detection = _det(nadir_camera_intrinsics.principal_point_px)

        tagged = tag_single(
            detection,
            nadir_telemetry,
            nadir_camera_intrinsics,
            rigid_mount_extrinsics,
            ground_elevation_msl=100.0,
        )

        assert tagged.latitude == pytest.approx(nadir_telemetry.latitude, abs=1e-6)
        assert tagged.longitude == pytest.approx(nadir_telemetry.longitude, abs=1e-6)
        assert tagged.altitude == pytest.approx(100.0, abs=1e-4)

    def test_preserves_detection_fields(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        detection = _det(nadir_camera_intrinsics.principal_point_px)
        tagged = tag_single(
            detection, nadir_telemetry, nadir_camera_intrinsics, rigid_mount_extrinsics
        )
        assert tagged.class_name == detection.class_name
        assert tagged.confidence == detection.confidence
        assert tagged.bbox == detection.bbox
        assert tagged.pixel_center == detection.pixel_center
        assert tagged.timestamp == detection.timestamp

    def test_horizon_detection_raises_geotag_error(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        # far top-of-frame pixel under a straight-down gimbal can still graze
        # the horizon depending on FOV; force it by using a body-level (non-gimbal)
        # telemetry sample with a pixel far off-center instead.
        level_telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        detection = _det((960.0, 540.0))  # camera looking straight forward, not down
        with pytest.raises(GeoTagError):
            tag_single(detection, level_telemetry, nadir_camera_intrinsics, rigid_mount_extrinsics)

    def test_stream_telemetry_source_is_time_synced(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        stream = TelemetryStream([nadir_telemetry])
        detection = _det(
            nadir_camera_intrinsics.principal_point_px, timestamp="2026-07-01T15:42:11Z"
        )
        tagged = tag_single(
            detection, stream, nadir_camera_intrinsics, rigid_mount_extrinsics, 100.0
        )
        assert tagged.latitude == pytest.approx(nadir_telemetry.latitude, abs=1e-6)

    def test_telemetry_too_far_raises(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        detection = _det(
            nadir_camera_intrinsics.principal_point_px, timestamp="2026-07-01T16:00:00Z"
        )
        with pytest.raises(TelemetryNotFoundError):
            tag_single(
                detection,
                [nadir_telemetry],
                nadir_camera_intrinsics,
                rigid_mount_extrinsics,
                100.0,
                max_telemetry_delta_seconds=5.0,
            )


class TestTagDetections:
    def test_batch_tags_all_detections(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        detections = [
            _det(nadir_camera_intrinsics.principal_point_px),
            _det((nadir_camera_intrinsics.principal_point_px[0] + 50, 540.0)),
        ]
        results = tag_detections(
            detections, nadir_telemetry, nadir_camera_intrinsics, rigid_mount_extrinsics, 100.0
        )
        assert len(results) == 2

    def test_skip_errors_true_drops_failures(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics
    ) -> None:
        level_telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        bad_detection = _det((960.0, 540.0))  # horizon-level, will fail to project
        results = tag_detections(
            [bad_detection],
            level_telemetry,
            nadir_camera_intrinsics,
            rigid_mount_extrinsics,
            skip_errors=True,
        )
        assert results == []

    def test_skip_errors_false_raises(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics
    ) -> None:
        level_telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        bad_detection = _det((960.0, 540.0))
        with pytest.raises(GeoTagError):
            tag_detections(
                [bad_detection],
                level_telemetry,
                nadir_camera_intrinsics,
                rigid_mount_extrinsics,
                skip_errors=False,
            )

    def test_empty_input_returns_empty(
        self, nadir_camera_intrinsics, rigid_mount_extrinsics, nadir_telemetry
    ) -> None:
        assert (
            tag_detections([], nadir_telemetry, nadir_camera_intrinsics, rigid_mount_extrinsics)
            == []
        )
