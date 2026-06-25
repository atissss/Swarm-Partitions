"""
tests/test_kml_loader.py
------------------------
Unit tests for kml_loader.py — updated for the bug-fixed version.

Key changes vs original tests:
  - Duplicate Boundary placemarks raise ValueError (BUG 8 fix).
  - Unnamed placemarks are skipped, not added as no-go zones (BUG 8 fix).
  - Boundary matching is case-insensitive (BUG 8 fix).
  - Non-polygon placemarks are skipped cleanly (BUG 8 fix).
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

VALID_KML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <Placemark>
          <name>Mission Boundary</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.100,28.700,0 77.200,28.700,0 77.200,28.800,0 77.100,28.800,0 77.100,28.700,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
        <Placemark>
          <name>NoGo_Lake</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.130,28.730,0 77.150,28.730,0 77.150,28.750,0 77.130,28.750,0 77.130,28.730,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
      </Document>
    </kml>
""")

# "boundary" is lowercase — tests case-insensitive matching (BUG 8 fix)
LOWERCASE_BOUNDARY_KML = VALID_KML.replace("Mission Boundary", "mission boundary")

DUPLICATE_BOUNDARY_KML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <Placemark>
          <name>Boundary A</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.100,28.700,0 77.200,28.700,0 77.200,28.800,0 77.100,28.700,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
        <Placemark>
          <name>Boundary B</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.110,28.710,0 77.190,28.710,0 77.190,28.790,0 77.110,28.710,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
      </Document>
    </kml>
""")

NO_BOUNDARY_KML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <Placemark>
          <name>SomeZone</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.100,28.700,0 77.200,28.700,0 77.200,28.800,0 77.100,28.700,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
      </Document>
    </kml>
""")

UNNAMED_NOGO_KML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <Placemark>
          <name>Boundary</name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.100,28.700,0 77.200,28.700,0 77.200,28.800,0 77.100,28.800,0 77.100,28.700,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
        <Placemark>
          <name></name>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
            77.130,28.730,0 77.150,28.730,0 77.150,28.750,0 77.130,28.730,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>
      </Document>
    </kml>
""")


def _tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".kml", delete=False, mode="w")
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadKml:
    def test_returns_expected_keys(self):
        result = load_kml(_tmp(VALID_KML))
        assert set(result.keys()) == {"boundary", "nogo", "epsg_code", "to_meters", "to_latlon"}

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

    def test_epsg_code_format(self):
        result = load_kml(_tmp(VALID_KML))
        assert result["epsg_code"].startswith("epsg:")

    def test_raises_if_no_boundary(self):
        with pytest.raises(ValueError, match="[Bb]oundary"):
            load_kml(_tmp(NO_BOUNDARY_KML))

    def test_case_insensitive_boundary_detection(self):
        """BUG 8 FIX: 'boundary' in lowercase should still be detected."""
        result = load_kml(_tmp(LOWERCASE_BOUNDARY_KML))
        assert result["boundary"] is not None

    def test_raises_on_duplicate_boundary(self):
        """BUG 8 FIX: two boundary placemarks should raise, not silently overwrite."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_kml(_tmp(DUPLICATE_BOUNDARY_KML))

    def test_unnamed_placemark_is_skipped(self):
        """BUG 8 FIX: unnamed placemarks should not become no-go zones."""
        result = load_kml(_tmp(UNNAMED_NOGO_KML))
        assert len(result["nogo"]) == 0