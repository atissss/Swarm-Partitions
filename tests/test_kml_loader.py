"""
tests/test_kml_loader.py
------------------------
Unit tests for kml_loader.py — covers home point parsing in addition to
the existing boundary / no-go zone / validation tests.
"""

import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mission_partitioner.kml_loader import load_kml


# ---------------------------------------------------------------------------
# KML fixtures
# ---------------------------------------------------------------------------

def _kml(body: str) -> str:
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>{body}</Document>
        </kml>
    """)

def _polygon_pm(name: str, coords: str) -> str:
    return f"""
        <Placemark>
          <name>{name}</name>
          <Polygon><outerBoundaryIs><LinearRing>
            <coordinates>{coords}</coordinates>
          </LinearRing></outerBoundaryIs></Polygon>
        </Placemark>"""

def _point_pm(name: str, lon: float, lat: float) -> str:
    return f"""
        <Placemark>
          <name>{name}</name>
          <Point><coordinates>{lon},{lat},0</coordinates></Point>
        </Placemark>"""

BOUNDARY_COORDS = "77.100,28.700,0 77.200,28.700,0 77.200,28.800,0 77.100,28.800,0 77.100,28.700,0"
NOGO_COORDS     = "77.130,28.730,0 77.150,28.730,0 77.150,28.750,0 77.130,28.730,0"

VALID_KML = _kml(
    _polygon_pm("Mission Boundary", BOUNDARY_COORDS) +
    _polygon_pm("NoGo_Lake", NOGO_COORDS)
)

VALID_KML_WITH_HOME = _kml(
    _polygon_pm("Mission Boundary", BOUNDARY_COORDS) +
    _polygon_pm("NoGo_Lake", NOGO_COORDS) +
    _point_pm("Home", 77.150, 28.750)
)

# Home only (no other polygon yet) — tests that home can seed the projection
HOME_FIRST_KML = _kml(
    _point_pm("Home Base", 77.150, 28.750) +
    _polygon_pm("Boundary", BOUNDARY_COORDS)
)

DUPLICATE_HOME_KML = _kml(
    _polygon_pm("Boundary", BOUNDARY_COORDS) +
    _point_pm("Home", 77.150, 28.750) +
    _point_pm("Home Backup", 77.160, 28.760)
)

DUPLICATE_BOUNDARY_KML = _kml(
    _polygon_pm("Boundary A", BOUNDARY_COORDS) +
    _polygon_pm("Boundary B", NOGO_COORDS)
)

NO_BOUNDARY_KML = _kml(_polygon_pm("SomeZone", NOGO_COORDS))

UNNAMED_PM_KML = _kml(
    _polygon_pm("Boundary", BOUNDARY_COORDS) +
    _polygon_pm("", NOGO_COORDS)
)

LOWERCASE_BOUNDARY_KML = _kml(_polygon_pm("mission boundary", BOUNDARY_COORDS))


def _tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".kml", delete=False, mode="w")
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests — existing behaviour
# ---------------------------------------------------------------------------

class TestLoadKml:
    def test_returns_expected_keys(self):
        result = load_kml(_tmp(VALID_KML))
        assert set(result.keys()) == {"boundary", "nogo", "home", "epsg_code", "to_meters", "to_latlon"}

    def test_boundary_is_valid_polygon(self):
        result = load_kml(_tmp(VALID_KML))
        assert result["boundary"] is not None
        assert result["boundary"].is_valid
        assert result["boundary"].area > 0

    def test_nogo_parsed_correctly(self):
        result = load_kml(_tmp(VALID_KML))
        assert len(result["nogo"]) == 1
        name, poly = result["nogo"][0]
        assert "NoGo_Lake" in name
        assert poly.is_valid

    def test_home_is_none_when_absent(self):
        result = load_kml(_tmp(VALID_KML))
        assert result["home"] is None

    def test_epsg_code_format(self):
        result = load_kml(_tmp(VALID_KML))
        assert result["epsg_code"].startswith("epsg:")

    def test_case_insensitive_boundary(self):
        result = load_kml(_tmp(LOWERCASE_BOUNDARY_KML))
        assert result["boundary"] is not None

    def test_raises_on_duplicate_boundary(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_kml(_tmp(DUPLICATE_BOUNDARY_KML))

    def test_raises_if_no_boundary(self):
        with pytest.raises(ValueError, match="[Bb]oundary"):
            load_kml(_tmp(NO_BOUNDARY_KML))

    def test_unnamed_placemark_skipped(self):
        result = load_kml(_tmp(UNNAMED_PM_KML))
        assert len(result["nogo"]) == 0


# ---------------------------------------------------------------------------
# Tests — home point
# ---------------------------------------------------------------------------

class TestHomePoint:
    def test_home_parsed_correctly(self):
        result = load_kml(_tmp(VALID_KML_WITH_HOME))
        assert result["home"] is not None
        lon, lat = result["home"]
        assert lon == pytest.approx(77.150, abs=1e-6)
        assert lat == pytest.approx(28.750, abs=1e-6)

    def test_home_is_wgs84_tuple(self):
        result = load_kml(_tmp(VALID_KML_WITH_HOME))
        assert isinstance(result["home"], tuple)
        assert len(result["home"]) == 2

    def test_home_longitude_in_range(self):
        result = load_kml(_tmp(VALID_KML_WITH_HOME))
        lon, lat = result["home"]
        assert -180 <= lon <= 180
        assert -90  <= lat <= 90

    def test_home_can_seed_projection(self):
        """Home point appearing before any polygon should still set up the CRS."""
        result = load_kml(_tmp(HOME_FIRST_KML))
        assert result["epsg_code"] is not None
        assert result["boundary"] is not None

    def test_raises_on_duplicate_home(self):
        with pytest.raises(ValueError, match="[Dd]uplicate Home"):
            load_kml(_tmp(DUPLICATE_HOME_KML))