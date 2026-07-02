"""
tests/test_exporter.py
----------------------
Unit tests for exporter.py — includes home point serialisation tests.
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

BOUNDARY      = Polygon([(300000, 3200000), (301000, 3200000), (301000, 3201000), (300000, 3201000)])
PART_1        = Polygon([(300000, 3200000), (300500, 3200000), (300500, 3201000), (300000, 3201000)])
PART_2        = Polygon([(300500, 3200000), (301000, 3200000), (301000, 3201000), (300500, 3201000)])
NOGO          = Polygon([(300100, 3200100), (300200, 3200100), (300200, 3200200), (300100, 3200200)])
POLY_WITH_HOLE = Polygon(
    [(300000, 3200000), (301000, 3200000), (301000, 3201000), (300000, 3201000)],
    [[(300200, 3200200), (300800, 3200200), (300800, 3200800), (300200, 3200800)]],
)
MULTI = MultiPolygon([
    Polygon([(300000, 3200000), (300400, 3200000), (300400, 3200400), (300000, 3200400)]),
    Polygon([(300600, 3200600), (301000, 3200600), (301000, 3201000), (300600, 3201000)]),
])

HOME_LONLAT = (77.15, 28.75)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _export(tmp_path: Path, **overrides) -> dict:
    defaults = dict(
        boundary=BOUNDARY,
        partitions=[PART_1, PART_2],
        predetermined_nogo=[("TestNogo", NOGO)],
        dynamic_nogo=[],
        to_latlon=TO_LATLON,
        epsg_code=EPSG,
        home=None,
        random_seed=42,
        output_path=tmp_path / "out.json",
    )
    defaults.update(overrides)
    export_json(**defaults)
    return json.loads((tmp_path / "out.json").read_text())


# ---------------------------------------------------------------------------
# poly_to_latlon tests
# ---------------------------------------------------------------------------

class TestPolyToLatlon:
    def test_simple_polygon_one_ring(self):
        result = poly_to_latlon(BOUNDARY, TO_LATLON)
        assert len(result) == 1
        assert "exterior" in result[0]
        assert "holes" in result[0]

    def test_coordinates_in_wgs84_range(self):
        result = poly_to_latlon(BOUNDARY, TO_LATLON)
        for lon, lat in result[0]["exterior"]:
            assert -180 <= lon <= 180
            assert -90  <= lat <= 90

    def test_simple_polygon_no_holes(self):
        assert poly_to_latlon(BOUNDARY, TO_LATLON)[0]["holes"] == []

    def test_polygon_with_hole_exports_hole(self):
        result = poly_to_latlon(POLY_WITH_HOLE, TO_LATLON)
        assert len(result[0]["holes"]) == 1

    def test_multipolygon_multiple_rings(self):
        result = poly_to_latlon(MULTI, TO_LATLON)
        assert len(result) == 2

    def test_empty_polygon_returns_empty_list(self):
        assert poly_to_latlon(Polygon(), TO_LATLON) == []

    def test_none_returns_empty_list(self):
        assert poly_to_latlon(None, TO_LATLON) == []


# ---------------------------------------------------------------------------
# export_json structure tests
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_file_created(self, tmp_path):
        out = tmp_path / "out.json"
        export_json(
            boundary=BOUNDARY, partitions=[PART_1],
            predetermined_nogo=[], dynamic_nogo=[],
            to_latlon=TO_LATLON, epsg_code=EPSG,
            home=None, random_seed=42, output_path=out,
        )
        assert out.exists()

    def test_top_level_keys(self, tmp_path):
        data = _export(tmp_path)
        assert set(data.keys()) == {"metadata", "home", "boundary", "partitions", "no_go_zones"}

    def test_metadata_crs_structure(self, tmp_path):
        crs = _export(tmp_path)["metadata"]["crs"]
        assert crs["coordinates"] == "EPSG:4326"
        assert crs["axis_order"] == ["longitude", "latitude"]
        assert "planning" in crs

    def test_metadata_generation_seed(self, tmp_path):
        assert _export(tmp_path)["metadata"]["generation"]["random_seed"] == 42

    def test_partition_ids_sequential(self, tmp_path):
        ids = [p["id"] for p in _export(tmp_path)["partitions"]]
        assert ids == [1, 2]

    def test_partition_uses_geometry_key(self, tmp_path):
        for p in _export(tmp_path)["partitions"]:
            assert "geometry" in p

    def test_predetermined_nogo_present(self, tmp_path):
        zones = _export(tmp_path)["no_go_zones"]["predetermined"]
        assert len(zones) == 1
        assert zones[0]["name"] == "TestNogo"

    def test_dynamic_nogo_empty(self, tmp_path):
        assert _export(tmp_path)["no_go_zones"]["dynamic"] == []

    def test_multipolygon_exported(self, tmp_path):
        data = _export(tmp_path, partitions=[MULTI])
        assert len(data["partitions"][0]["geometry"]) == 2

    def test_hole_exported(self, tmp_path):
        data = _export(tmp_path, partitions=[POLY_WITH_HOLE])
        assert len(data["partitions"][0]["geometry"][0]["holes"]) == 1


# ---------------------------------------------------------------------------
# Home point tests
# ---------------------------------------------------------------------------

class TestHomePoint:
    def test_home_null_when_not_provided(self, tmp_path):
        data = _export(tmp_path, home=None)
        assert data["home"] is None

    def test_home_present_when_provided(self, tmp_path):
        data = _export(tmp_path, home=HOME_LONLAT)
        assert data["home"] is not None

    def test_home_has_longitude_and_latitude_keys(self, tmp_path):
        home = _export(tmp_path, home=HOME_LONLAT)["home"]
        assert "longitude" in home
        assert "latitude"  in home

    def test_home_values_match_input(self, tmp_path):
        home = _export(tmp_path, home=HOME_LONLAT)["home"]
        assert home["longitude"] == pytest.approx(HOME_LONLAT[0], abs=1e-8)
        assert home["latitude"]  == pytest.approx(HOME_LONLAT[1], abs=1e-8)

    def test_home_is_top_level_key(self, tmp_path):
        """Home should be directly accessible at the top level of the JSON."""
        data = _export(tmp_path, home=HOME_LONLAT)
        assert "home" in data
        assert data["home"]["longitude"] == pytest.approx(HOME_LONLAT[0])