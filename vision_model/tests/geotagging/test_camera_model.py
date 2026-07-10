from __future__ import annotations

import numpy as np
import pytest

from vision_model.geotagging.camera_model import pixel_to_ray


class TestPixelToRay:
    def test_principal_point_maps_to_pure_forward_ray(self, nadir_camera_intrinsics) -> None:
        cx, cy = nadir_camera_intrinsics.principal_point_px
        ray = pixel_to_ray((cx, cy), nadir_camera_intrinsics)
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)

    def test_ray_is_unit_length(self, nadir_camera_intrinsics) -> None:
        ray = pixel_to_ray((1200.0, 700.0), nadir_camera_intrinsics)
        assert np.linalg.norm(ray) == pytest.approx(1.0)

    def test_pixel_right_of_center_has_positive_x(self, nadir_camera_intrinsics) -> None:
        cx, cy = nadir_camera_intrinsics.principal_point_px
        ray = pixel_to_ray((cx + 100, cy), nadir_camera_intrinsics)
        assert ray[0] > 0
        assert ray[1] == pytest.approx(0.0, abs=1e-9)

    def test_pixel_below_center_has_positive_y(self, nadir_camera_intrinsics) -> None:
        cx, cy = nadir_camera_intrinsics.principal_point_px
        ray = pixel_to_ray((cx, cy + 100), nadir_camera_intrinsics)
        assert ray[1] > 0
        assert ray[0] == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_pixels_give_symmetric_rays(self, nadir_camera_intrinsics) -> None:
        cx, cy = nadir_camera_intrinsics.principal_point_px
        ray_left = pixel_to_ray((cx - 50, cy), nadir_camera_intrinsics)
        ray_right = pixel_to_ray((cx + 50, cy), nadir_camera_intrinsics)
        assert ray_left[0] == pytest.approx(-ray_right[0])
        assert ray_left[2] == pytest.approx(ray_right[2])
