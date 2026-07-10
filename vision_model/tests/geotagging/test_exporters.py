from __future__ import annotations

import csv
import json
from pathlib import Path

from vision_model.geotagging.exporters import export_all, export_csv, export_geojson, export_json
from vision_model.interfaces.geotag import GeoTaggedDetection


def _tagged(class_name="car", lat=30.352814, lon=76.364921) -> GeoTaggedDetection:
    return GeoTaggedDetection(
        class_name=class_name,
        confidence=0.97,
        bbox=(10.0, 10.0, 60.0, 90.0),
        pixel_center=(35.0, 50.0),
        latitude=lat,
        longitude=lon,
        altitude=18.2,
        timestamp="2026-07-01T15:42:11Z",
    )


class TestExportJson:
    def test_writes_valid_json_list(self, tmp_path: Path) -> None:
        path = export_json([_tagged(), _tagged(class_name="person")], tmp_path / "out.json")
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["class_name"] == "car"
        assert data[1]["class_name"] == "person"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = export_json([_tagged()], tmp_path / "nested" / "dir" / "out.json")
        assert path.is_file()


class TestExportCsv:
    def test_writes_valid_csv_with_header(self, tmp_path: Path) -> None:
        path = export_csv([_tagged()], tmp_path / "out.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["class_name"] == "car"
        assert float(rows[0]["latitude"]) == 30.352814
        assert float(rows[0]["altitude"]) == 18.2

    def test_empty_list_writes_header_only(self, tmp_path: Path) -> None:
        path = export_csv([], tmp_path / "empty.csv")
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows == []


class TestExportGeojson:
    def test_writes_valid_feature_collection(self, tmp_path: Path) -> None:
        path = export_geojson([_tagged(), _tagged()], tmp_path / "out.geojson")
        data = json.loads(path.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2
        assert data["features"][0]["geometry"]["type"] == "Point"
        assert data["features"][0]["geometry"]["coordinates"][:2] == [76.364921, 30.352814]


class TestExportAll:
    def test_writes_all_three_formats(self, tmp_path: Path) -> None:
        paths = export_all([_tagged()], tmp_path / "exports", basename="flight1")
        assert set(paths) == {"json", "csv", "geojson"}
        for p in paths.values():
            assert p.is_file()
        assert paths["json"].name == "flight1.json"
        assert paths["csv"].name == "flight1.csv"
        assert paths["geojson"].name == "flight1.geojson"
