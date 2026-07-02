"""
tests/test_partitioner.py
-------------------------
Unit tests for partitioner.py — includes tests for small area and thin strip
filtering (min_area_m2 and min_width_m parameters).
"""

import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mission_partitioner.partitioner import _is_usable, build_partitions

# 1 km × 1 km square in UTM metres
SQUARE = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])

# Small obstacle in one corner
OBSTACLE = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])


# ---------------------------------------------------------------------------
# _is_usable unit tests
# ---------------------------------------------------------------------------

class TestIsUsable:
    def test_large_square_passes(self):
        big = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # 10 000 m²
        assert _is_usable(big, min_area_m2=500, min_width_m=10) is True

    def test_tiny_area_fails(self):
        tiny = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])  # 25 m²
        assert _is_usable(tiny, min_area_m2=500, min_width_m=10) is False

    def test_thin_strip_fails_width_check(self):
        # 1 m wide × 2000 m long = 2000 m² (passes area), but only 1 m wide (fails width)
        strip = Polygon([(0, 0), (1, 0), (1, 2000), (0, 2000)])
        assert _is_usable(strip, min_area_m2=500, min_width_m=10) is False

    def test_wide_rectangle_passes(self):
        # 50 m × 50 m = 2500 m², 50 m wide — passes both checks
        rect = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        assert _is_usable(rect, min_area_m2=500, min_width_m=10) is True

    def test_area_threshold_boundary(self):
        # Exactly at the threshold — should pass (>=, not >)
        exact = Polygon([(0, 0), (500**0.5, 0), (500**0.5, 500**0.5), (0, 500**0.5)])
        # area ≈ 500 m²; allow slight float tolerance
        assert _is_usable(exact, min_area_m2=exact.area, min_width_m=1) is True

    def test_zero_width_threshold_ignores_width(self):
        # min_width_m=0 means no width check — thin strip should pass if area is ok
        strip = Polygon([(0, 0), (1, 0), (1, 2000), (0, 2000)])
        assert _is_usable(strip, min_area_m2=500, min_width_m=0) is True


# ---------------------------------------------------------------------------
# build_partitions integration tests
# ---------------------------------------------------------------------------

class TestBuildPartitions:
    def test_returns_correct_count(self):
        parts = build_partitions(SQUARE, [], n_parts=3, n_samples=500)
        assert len(parts) == 3

    def test_partitions_are_non_empty(self):
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        for p in parts:
            assert p is not None
            assert not p.is_empty

    def test_partitions_within_boundary(self):
        parts = build_partitions(SQUARE, [], n_parts=3, n_samples=500)
        expanded = SQUARE.buffer(1)
        for p in parts:
            assert expanded.contains(p) or expanded.covers(p)

    def test_no_significant_overlap(self):
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        for i, a in enumerate(parts):
            for j, b in enumerate(parts):
                if i >= j:
                    continue
                assert a.intersection(b).area < 1.0

    def test_nogo_zones_excluded(self):
        parts = build_partitions(SQUARE, [OBSTACLE], n_parts=3, n_samples=500)
        for p in parts:
            assert not OBSTACLE.contains(p.centroid)

    def test_single_partition_covers_boundary(self):
        parts = build_partitions(SQUARE, [], n_parts=1, n_samples=500)
        assert len(parts) == 1
        assert parts[0].area == pytest.approx(SQUARE.area, rel=0.01)

    def test_deterministic_with_same_seed(self):
        a = build_partitions(SQUARE, [], n_parts=3, n_samples=500, random_state=7)
        b = build_partitions(SQUARE, [], n_parts=3, n_samples=500, random_state=7)
        for pa, pb in zip(a, b):
            assert pa.centroid.x == pytest.approx(pb.centroid.x, rel=1e-6)
            assert pa.centroid.y == pytest.approx(pb.centroid.y, rel=1e-6)

    def test_partitions_sorted_by_centroid(self):
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        centroids = [(p.centroid.x, p.centroid.y) for p in parts]
        assert centroids == sorted(centroids)

    def test_raises_on_max_attempts_exceeded(self):
        with pytest.raises(RuntimeError, match="max_attempts"):
            build_partitions(SQUARE, [], n_parts=2, n_samples=500, max_attempts=1)

    def test_min_area_default_accepted(self):
        """Default thresholds should not discard normal partitions of a 1 km² square."""
        parts = build_partitions(
            SQUARE, [], n_parts=3, n_samples=500,
            min_area_m2=500, min_width_m=10,
        )
        assert len(parts) == 3

    def test_very_high_min_area_raises(self):
        """Threshold so high that all fragments are discarded → RuntimeError."""
        with pytest.raises(RuntimeError):
            build_partitions(
                SQUARE, [], n_parts=3, n_samples=500,
                min_area_m2=9_999_999,  # larger than the whole boundary
                min_width_m=0,
            )

    def test_custom_min_area_and_width_accepted(self):
        """Reasonable custom thresholds should still produce n_parts."""
        parts = build_partitions(
            SQUARE, [], n_parts=2, n_samples=500,
            min_area_m2=100, min_width_m=5,
        )
        assert len(parts) == 2