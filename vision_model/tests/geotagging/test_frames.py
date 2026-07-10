from __future__ import annotations

import numpy as np
import pytest

from vision_model.geotagging.frames import (
    camera_ray_to_ned,
    euler_to_rotation_matrix,
    ned_offset_to_geodetic,
)
from vision_model.interfaces.camera import CameraExtrinsics
from vision_model.interfaces.telemetry import Telemetry


class TestEulerToRotationMatrix:
    def test_zero_angles_is_identity(self) -> None:
        r = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-9)

    def test_is_orthonormal(self) -> None:
        r = euler_to_rotation_matrix(37.0, -12.0, 5.0)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_yaw_90_maps_forward_to_east(self) -> None:
        # forward = body x-axis = [1, 0, 0]; yaw=90 should rotate it to east = [0, 1, 0]
        r = euler_to_rotation_matrix(90.0, 0.0, 0.0)
        result = r @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-9)

    def test_pitch_minus_90_maps_forward_to_down(self) -> None:
        # forward tilted straight down: body x-axis -> NED down = [0, 0, 1]
        r = euler_to_rotation_matrix(0.0, -90.0, 0.0)
        result = r @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-9)


class TestCameraRayToNed:
    def test_nadir_boresight_points_straight_down(self) -> None:
        telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            gimbal_yaw=0.0,
            gimbal_pitch=-90.0,
            gimbal_roll=0.0,
        )
        ray_camera = np.array([0.0, 0.0, 1.0])  # straight down the camera boresight
        ray_ned = camera_ray_to_ned(ray_camera, telemetry, CameraExtrinsics())
        np.testing.assert_allclose(ray_ned, [0.0, 0.0, 1.0], atol=1e-9)

    def test_output_is_unit_length(self) -> None:
        telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=15.0,
            pitch=-3.0,
            roll=2.0,
        )
        ray_camera = np.array([0.1, 0.05, 0.99])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        ray_ned = camera_ray_to_ned(ray_camera, telemetry, CameraExtrinsics())
        assert np.linalg.norm(ray_ned) == pytest.approx(1.0)

    def test_mount_rotation_offset_is_applied(self) -> None:
        # A camera mounted with a 90 degree yaw offset should behave like a
        # body attitude rotated by 90 degrees would, for a purely-forward ray.
        telemetry = Telemetry(
            timestamp="2026-07-01T15:42:11Z",
            latitude=30.0,
            longitude=76.0,
            altitude_msl=100.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
        )
        ray_camera = np.array([0.0, 0.0, 1.0])  # camera boresight
        no_offset = camera_ray_to_ned(ray_camera, telemetry, CameraExtrinsics())
        with_offset = camera_ray_to_ned(
            ray_camera, telemetry, CameraExtrinsics(rotation_deg=(90.0, 0.0, 0.0))
        )
        assert not np.allclose(no_offset, with_offset)


class TestNedOffsetToGeodetic:
    def test_zero_offset_returns_reference_point(self) -> None:
        lat, lon, alt = ned_offset_to_geodetic((0.0, 0.0, 0.0), 30.352814, 76.364921, 118.2)
        assert lat == pytest.approx(30.352814, abs=1e-9)
        assert lon == pytest.approx(76.364921, abs=1e-9)
        assert alt == pytest.approx(118.2, abs=1e-6)

    def test_positive_down_offset_reduces_altitude(self) -> None:
        _, _, alt = ned_offset_to_geodetic((0.0, 0.0, 18.2), 30.0, 76.0, 118.2)
        assert alt == pytest.approx(100.0, abs=1e-6)

    def test_north_offset_increases_latitude(self) -> None:
        lat, _, _ = ned_offset_to_geodetic((100.0, 0.0, 0.0), 30.0, 76.0, 100.0)
        assert lat > 30.0

    def test_east_offset_increases_longitude(self) -> None:
        _, lon, _ = ned_offset_to_geodetic((0.0, 100.0, 0.0), 30.0, 76.0, 100.0)
        assert lon > 76.0
