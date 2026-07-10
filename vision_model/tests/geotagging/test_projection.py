from __future__ import annotations

import numpy as np
import pytest

from vision_model.geotagging.projection import intersect_flat_ground
from vision_model.utils.exceptions import ProjectionError


class TestIntersectFlatGround:
    def test_straight_down_ray_lands_directly_below(self) -> None:
        ray = np.array([0.0, 0.0, 1.0])
        n, e, d = intersect_flat_ground(ray, height_above_ground_m=50.0)
        assert n == pytest.approx(0.0, abs=1e-9)
        assert e == pytest.approx(0.0, abs=1e-9)
        assert d == pytest.approx(50.0)

    def test_angled_ray_lands_offset_from_origin(self) -> None:
        ray = np.array([0.5, 0.0, 0.866])  # ~30 degrees off vertical, unit-ish
        ray = ray / np.linalg.norm(ray)
        n, e, d = intersect_flat_ground(ray, height_above_ground_m=100.0)
        assert d == pytest.approx(100.0)
        assert n > 0  # ray had a north component, so ground point is north of origin

    def test_horizon_ray_raises(self) -> None:
        ray = np.array([1.0, 0.0, 0.0])  # perfectly horizontal
        with pytest.raises(ProjectionError):
            intersect_flat_ground(ray, height_above_ground_m=50.0)

    def test_upward_ray_raises(self) -> None:
        ray = np.array([0.0, 0.0, -1.0])  # pointing up, away from ground
        with pytest.raises(ProjectionError):
            intersect_flat_ground(ray, height_above_ground_m=50.0)

    def test_nonpositive_height_raises(self) -> None:
        ray = np.array([0.0, 0.0, 1.0])
        with pytest.raises(ProjectionError):
            intersect_flat_ground(ray, height_above_ground_m=0.0)
        with pytest.raises(ProjectionError):
            intersect_flat_ground(ray, height_above_ground_m=-5.0)
