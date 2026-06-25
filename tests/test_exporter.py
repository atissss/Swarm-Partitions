"""
tests/test_exporter.py
----------------------
Unit tests for exporter.py — updated for the bug-fixed version.

Key changes vs original tests:
  - poly_to_latlon now returns list[dict] (with "exterior" and "holes" keys)
    instead of list[list[float]], so assertions check that structure.
  - export_json payload uses "geometry" key instead of "coords".
  - Metadata now has "crs" dict and "generation" dict instead of bare "epsg".
  - MultiPolygon export is tested (was previously untestable / would crash).
  - Hole (interior ring) export is tested.
"""

import json
import sys
from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mission_partitioner.exporter import export_json, poly_to_latlon

TO_LATLON = Transformer.from_crs("epsg:32643", "epsg:4326", always_xy=True)
EPSG = "epsg:32643"

BOUNDARY  = Polygon([(300000, 3200000), (301000, 3200000), (301000, 3201000), (300000, 3201000)])
PART_1    = Polygon([(300000, 3200000), (300500, 3200000), (300500, 3201000), (300000, 3201000)])
PART_2    = Polygon([(300500, 3200000), (301000, 3200000), (301000, 3201000), (300500, 3201000)])
NOGO      = Polygon([(300100, 3200100), (300200, 3200100), (300200, 3200200), (300100, 3200200)])

# Polygon with a hole: outer square minus inner square
OUTER     = Polygon([(300000, 3200000), (301000, 3200000), (301000, 3201000), (300000, 3201000)])
INNER     = [(300200, 3200200), (300800, 3200200), (300800, 3200800), (300200, 3200800)]
POLY_WITH_HOLE = Polygon(OUTER.exterior.coords, [INNER])

# Two disjoint squares → MultiPolygon
MULTI = MultiPolygon([
    Polygon([(300000, 3200000), (300400, 3200000), (300400, 3200400), (300000, 3200400)]),
    Polygon([(300600, 3200600), (301000, 3200600), (301000, 3201000), (300600, 3201000)]),
])


class TestPolyToLatlon:
    def test_simple_polygon_returns_one_ring_dict(self):
        result = poly_to_latlon(BOUNDARY, TO_LATLON)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "exterior" in result[0]
        assert "holes" in result[0]

    def test_exterior_contains_lonlat_pairs(self):
        result = poly_to_latlon(BOUNDARY, TO_LATLON)
        for lon, lat in result[0]["exterior"]:
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90

    def test_simple_polygon_has_no_holes(self):
        result = poly_to_latlon(BOUNDARY, TO_LATLON)
        assert result[0]["holes"] == []

    def test_polygon_with_hole_exports_hole(self):
        result = poly_to_latlon(POLY_WITH_HOLE, TO_LATLON)
        assert len(result) == 1
        assert len(result[0]["holes"]) == 1
        assert len(result[0]["holes"][0]) > 0

    def test_multipolygon_returns_multiple_ring_dicts(self):
        result = poly_to_latlon(MULTI, TO_LATLON)
        assert len(result) == 2
        for entry in result:
            assert "exterior" in entry
            assert "holes" in entry

    def test_empty_polygon_returns_empty_list(self):
        assert poly_to_latlon(Polygon(), TO_LATLON) == []

    def test_none_returns_empty_list(self):
        assert poly_to_latlon(None, TO_LATLON) == []


class TestExportJson:
    def _export(self, tmp_path: Path, **overrides) -> dict:
        defaults = dict(
            boundary=BOUNDARY,
            partitions=[PART_1, PART_2],
            predetermined_nogo=[("TestNogo", NOGO)],
            dynamic_nogo=[],
            to_latlon=TO_LATLON,
            epsg_code=EPSG,
            random_seed=42,
            output_path=tmp_path / "out.json",
        )
        defaults.update(overrides)
        export_json(**defaults)
        return json.loads((tmp_path / "out.json").read_text())

    def test_file_is_created(self, tmp_path):
        out = tmp_path / "output.json"
        export_json(
            boundary=BOUNDARY, partitions=[PART_1],
            predetermined_nogo=[], dynamic_nogo=[],
            to_latlon=TO_LATLON, epsg_code=EPSG,
            random_seed=42, output_path=out,
        )
        assert out.exists()

    def test_top_level_keys(self, tmp_path):
        data = self._export(tmp_path)
        assert set(data.keys()) == {"metadata", "boundary", "partitions", "no_go_zones"}

    def test_metadata_crs_structure(self, tmp_path):
        """BUG 3 FIX: CRS is no longer ambiguous."""
        data = self._export(tmp_path)
        crs = data["metadata"]["crs"]
        assert crs["coordinates"] == "EPSG:4326"
        assert crs["axis_order"] == ["longitude", "latitude"]
        assert "planning" in crs

    def test_metadata_generation_seed(self, tmp_path):
        """BUG 4 FIX: seed stored in metadata."""
        data = self._export(tmp_path)
        assert data["metadata"]["generation"]["random_seed"] == 42

    def test_partition_count_in_metadata(self, tmp_path):
        data = self._export(tmp_path)
        assert data["metadata"]["n_partitions"] == 2

    def test_partition_ids_sequential(self, tmp_path):
        data = self._export(tmp_path)
        ids = [p["id"] for p in data["partitions"]]
        assert ids == [1, 2]

    def test_partition_geometry_key(self, tmp_path):
        """Partitions now use 'geometry' not 'coords'."""
        data = self._export(tmp_path)
        for p in data["partitions"]:
            assert "geometry" in p
            assert isinstance(p["geometry"], list)

    def test_predetermined_nogo_present(self, tmp_path):
        data = self._export(tmp_path)
        zones = data["no_go_zones"]["predetermined"]
        assert len(zones) == 1
        assert zones[0]["name"] == "TestNogo"
        assert len(zones[0]["geometry"]) > 0

    def test_dynamic_nogo_empty_list(self, tmp_path):
        data = self._export(tmp_path)
        assert data["no_go_zones"]["dynamic"] == []

    def test_dynamic_nogo_with_polygon(self, tmp_path):
        data = self._export(tmp_path, dynamic_nogo=[NOGO])
        assert len(data["no_go_zones"]["dynamic"]) == 1
        assert data["no_go_zones"]["dynamic"][0]["id"] == 1

    def test_multipolygon_partition_exported(self, tmp_path):
        """BUG 2 FIX: MultiPolygon partitions no longer crash."""
        data = self._export(tmp_path, partitions=[MULTI])
        geom = data["partitions"][0]["geometry"]
        assert len(geom) == 2  # two sub-polygons

    def test_polygon_with_hole_exported(self, tmp_path):
        """BUG 1 FIX: holes are exported."""
        data = self._export(tmp_path, partitions=[POLY_WITH_HOLE])
        geom = data["partitions"][0]["geometry"]
        assert len(geom[0]["holes"]) == 1