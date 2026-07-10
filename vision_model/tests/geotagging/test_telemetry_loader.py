from __future__ import annotations

from pathlib import Path

import pytest

from vision_model.geotagging.telemetry_loader import TelemetryStream, load_telemetry_csv
from vision_model.utils.exceptions import TelemetryNotFoundError, VisionModelError

_CSV_HEADER = "timestamp,latitude,longitude,altitude_msl,yaw,pitch,roll,gimbal_pitch\n"
_CSV_ROWS = (
    "2026-07-01T15:42:10Z,30.0000,76.0000,118.0,0,0,0,-90\n"
    "2026-07-01T15:42:11Z,30.0001,76.0001,118.2,1,0,0,-90\n"
    "2026-07-01T15:42:12Z,30.0002,76.0002,118.4,2,0,0,-90\n"
)


@pytest.fixture
def telemetry_csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "flight.csv"
    p.write_text(_CSV_HEADER + _CSV_ROWS)
    return p


class TestLoadTelemetryCsv:
    def test_parses_all_rows_sorted(self, telemetry_csv_path: Path) -> None:
        samples = load_telemetry_csv(telemetry_csv_path)
        assert len(samples) == 3
        assert [s.timestamp for s in samples] == sorted(s.timestamp for s in samples)

    def test_optional_gimbal_columns_present(self, telemetry_csv_path: Path) -> None:
        samples = load_telemetry_csv(telemetry_csv_path)
        assert samples[0].gimbal_pitch == -90.0
        assert samples[0].gimbal_yaw is None  # not in this fixture's header

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VisionModelError):
            load_telemetry_csv(tmp_path / "nope.csv")

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("timestamp,latitude,longitude\n2026-07-01T15:42:11Z,30.0,76.0\n")
        with pytest.raises(VisionModelError):
            load_telemetry_csv(p)

    def test_empty_data_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text(_CSV_HEADER)
        with pytest.raises(TelemetryNotFoundError):
            load_telemetry_csv(p)

    def test_malformed_row_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "malformed.csv"
        p.write_text(_CSV_HEADER + "2026-07-01T15:42:10Z,not_a_number,76.0,118.0,0,0,0,-90\n")
        with pytest.raises(VisionModelError):
            load_telemetry_csv(p)


class TestTelemetryStream:
    def test_from_csv_and_nearest(self, telemetry_csv_path: Path) -> None:
        stream = TelemetryStream.from_csv(telemetry_csv_path)
        assert len(stream) == 3
        nearest = stream.nearest("2026-07-01T15:42:11Z")
        assert nearest.latitude == pytest.approx(30.0001)

    def test_nearest_respects_max_delta(self, telemetry_csv_path: Path) -> None:
        stream = TelemetryStream.from_csv(telemetry_csv_path)
        with pytest.raises(TelemetryNotFoundError):
            stream.nearest("2026-07-01T16:00:00Z", max_delta_seconds=5.0)

    def test_empty_samples_raises(self) -> None:
        with pytest.raises(TelemetryNotFoundError):
            TelemetryStream([])
