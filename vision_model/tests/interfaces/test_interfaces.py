from __future__ import annotations

import pytest

from vision_model.interfaces.camera import CameraExtrinsics, CameraIntrinsics
from vision_model.interfaces.detection import Detection
from vision_model.interfaces.geotag import GeoTaggedDetection
from vision_model.interfaces.telemetry import Telemetry


class TestDetection:
    def test_valid_detection_constructs(self) -> None:
        d = Detection(
            class_name="car",
            confidence=0.97,
            bbox=(10.0, 10.0, 60.0, 90.0),
            pixel_center=(35.0, 50.0),
            frame_id="frame_0001",
            timestamp="2026-07-01T15:42:11Z",
        )
        assert d.width == 50.0
        assert d.height == 80.0
        assert d.area == 4000.0

    def test_is_frozen(self) -> None:
        d = Detection(
            class_name="car",
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            pixel_center=(5, 5),
            frame_id="f",
            timestamp="2026-07-01T15:42:11Z",
        )
        with pytest.raises(AttributeError):
            d.confidence = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.1])
    def test_rejects_out_of_range_confidence(self, bad_conf: float) -> None:
        with pytest.raises(ValueError):
            Detection(
                class_name="person",
                confidence=bad_conf,
                bbox=(0, 0, 10, 10),
                pixel_center=(5, 5),
                frame_id="f",
                timestamp="2026-07-01T15:42:11Z",
            )

    def test_rejects_degenerate_bbox(self) -> None:
        with pytest.raises(ValueError):
            Detection(
                class_name="person",
                confidence=0.5,
                bbox=(10, 10, 10, 20),  # x1 == x2
                pixel_center=(10, 15),
                frame_id="f",
                timestamp="2026-07-01T15:42:11Z",
            )


class TestTelemetry:
    def test_effective_attitude_prefers_gimbal(self) -> None:
        t = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=10.0,
            pitch=5.0,
            roll=1.0,
            gimbal_yaw=0.0,
            gimbal_pitch=-90.0,
            gimbal_roll=0.0,
        )
        assert t.effective_yaw == 0.0
        assert t.effective_pitch == -90.0
        assert t.effective_roll == 0.0

    def test_effective_attitude_falls_back_to_body(self) -> None:
        t = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=10.0,
            pitch=5.0,
            roll=1.0,
        )
        assert t.effective_yaw == 10.0
        assert t.effective_pitch == 5.0
        assert t.effective_roll == 1.0

    @pytest.mark.parametrize("bad_lat", [-91.0, 91.0])
    def test_rejects_invalid_latitude(self, bad_lat: float) -> None:
        with pytest.raises(ValueError):
            Telemetry(
                timestamp="2026-07-01T15:42:11Z",
                latitude=bad_lat,
                longitude=0.0,
                altitude_msl=100.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
            )


class TestCamera:
    def test_intrinsics_rejects_nonpositive_focal_length(self) -> None:
        with pytest.raises(ValueError):
            CameraIntrinsics(
                focal_length_px=(0.0, 1000.0),
                principal_point_px=(960.0, 540.0),
                image_size_px=(1920, 1080),
            )

    def test_extrinsics_defaults_to_rigid_mount(self) -> None:
        e = CameraExtrinsics()
        assert e.translation_m == (0.0, 0.0, 0.0)
        assert e.rotation_deg == (0.0, 0.0, 0.0)


class TestGeoTaggedDetection:
    def test_serializes_to_geojson_feature(self) -> None:
        g = GeoTaggedDetection(
            class_name="car",
            confidence=0.97,
            bbox=(10.0, 10.0, 60.0, 90.0),
            pixel_center=(35.0, 50.0),
            latitude=30.352814,
            longitude=76.364921,
            altitude=18.2,
            timestamp="2026-07-01T15:42:11Z",
        )
        feature = g.to_geojson_feature()
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert feature["geometry"]["coordinates"] == [76.364921, 30.352814, 18.2]
        assert feature["properties"]["class_name"] == "car"

    def test_rejects_invalid_longitude(self) -> None:
        with pytest.raises(ValueError):
            GeoTaggedDetection(
                class_name="car",
                confidence=0.9,
                bbox=(0, 0, 10, 10),
                pixel_center=(5, 5),
                latitude=0.0,
                longitude=181.0,
                altitude=10.0,
                timestamp="2026-07-01T15:42:11Z",
            )
