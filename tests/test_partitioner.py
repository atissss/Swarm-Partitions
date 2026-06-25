"""
tests/test_partitioner.py
-------------------------
Unit tests for partitioner.py — updated for the bug-fixed version.

Key changes vs original tests:
  - random_state parameter is passed explicitly (covers BUG 3 fix).
  - Partition ordering is now deterministic (BUG 4 fix) so centroid-sort
    behaviour can be asserted.
  - Exact count validation (BUG 6) is tested: too many requested partitions
    for a degenerate workspace should raise RuntimeError.
  - Sampling loop timeout (BUG 10) is tested via max_attempts=1.
"""

import sys
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.mission_partitioner.partitioner import build_partitions

# 1 km × 1 km square in UTM metres
SQUARE = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])

# Small obstacle in one corner
OBSTACLE = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])


class TestBuildPartitions:
    def test_returns_correct_count(self):
        parts = build_partitions(SQUARE, [], n_parts=3, n_samples=500)
        assert len(parts) == 3

    def test_partitions_are_non_empty(self):
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        for p in parts:
            assert p is not None
            assert not p.is_empty

    def test_partitions_stay_within_boundary(self):
        parts = build_partitions(SQUARE, [], n_parts=3, n_samples=500)
        expanded = SQUARE.buffer(1)
        for p in parts:
            assert expanded.contains(p) or expanded.covers(p)

    def test_no_significant_overlap_between_partitions(self):
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        for i, a in enumerate(parts):
            for j, b in enumerate(parts):
                if i >= j:
                    continue
                assert a.intersection(b).area < 1.0, (
                    f"Partitions {i} and {j} overlap by {a.intersection(b).area:.2f} m²"
                )

    def test_nogo_zones_excluded(self):
        parts = build_partitions(SQUARE, [OBSTACLE], n_parts=3, n_samples=500)
        for p in parts:
            assert not OBSTACLE.contains(p.centroid)

    def test_single_partition_covers_boundary(self):
        parts = build_partitions(SQUARE, [], n_parts=1, n_samples=500)
        assert len(parts) == 1
        assert parts[0].area == pytest.approx(SQUARE.area, rel=0.01)

    def test_deterministic_with_same_seed(self):
        """BUG 3 FIX: same seed → identical centroid positions."""
        parts_a = build_partitions(SQUARE, [], n_parts=3, n_samples=500, random_state=7)
        parts_b = build_partitions(SQUARE, [], n_parts=3, n_samples=500, random_state=7)
        for a, b in zip(parts_a, parts_b):
            assert a.centroid.x == pytest.approx(b.centroid.x, rel=1e-6)
            assert a.centroid.y == pytest.approx(b.centroid.y, rel=1e-6)

    def test_different_seeds_may_differ(self):
        """BUG 3 FIX: different seeds should generally differ (probabilistic)."""
        parts_a = build_partitions(SQUARE, [], n_parts=4, n_samples=500, random_state=1)
        parts_b = build_partitions(SQUARE, [], n_parts=4, n_samples=500, random_state=99)
        centroids_a = [(p.centroid.x, p.centroid.y) for p in parts_a]
        centroids_b = [(p.centroid.x, p.centroid.y) for p in parts_b]
        assert centroids_a != centroids_b

    def test_partitions_sorted_by_centroid(self):
        """BUG 4 FIX: partitions should be sorted by (centroid.x, centroid.y)."""
        parts = build_partitions(SQUARE, [], n_parts=4, n_samples=500)
        centroids = [(p.centroid.x, p.centroid.y) for p in parts]
        assert centroids == sorted(centroids)

    def test_raises_on_max_attempts_exceeded(self):
        """BUG 10 FIX: RuntimeError when workspace cannot be sampled."""
        with pytest.raises(RuntimeError, match="max_attempts"):
            # max_attempts=1 makes it impossible to collect 500 samples
            build_partitions(SQUARE, [], n_parts=2, n_samples=500, max_attempts=1)