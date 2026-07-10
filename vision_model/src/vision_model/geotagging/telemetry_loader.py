"""Flight-log parsing into a `Telemetry` stream, and time-sync lookup.

MVP ships a parser for a generic CSV schema (documented below) rather than
any single vendor's proprietary log format. Vendor-specific parsers (DJI
SRT/CSV, PX4 ULog, ArduPilot .bin, AU-AIR's own per-frame telemetry) are
extension points: each is a `path -> list[Telemetry]` function that can be
registered alongside `load_telemetry_csv` without changing `TelemetryStream`
or `geotagger.py`, exactly like `datasets.registry` for annotation formats.
"""

from __future__ import annotations

import csv
from pathlib import Path

from vision_model.interfaces.telemetry import Telemetry
from vision_model.utils.exceptions import TelemetryNotFoundError, VisionModelError
from vision_model.utils.logging_setup import get_logger
from vision_model.utils.time_sync import find_nearest

logger = get_logger(__name__)

# Required columns for the generic CSV schema. Optional gimbal_* columns may
# be omitted entirely (falls back to body attitude via Telemetry.effective_*).
_REQUIRED_COLUMNS = {
    "timestamp",
    "latitude",
    "longitude",
    "altitude_msl",
    "yaw",
    "pitch",
    "roll",
}
_OPTIONAL_COLUMNS = {"gimbal_yaw", "gimbal_pitch", "gimbal_roll"}


def load_telemetry_csv(path: str | Path) -> list[Telemetry]:
    """Parse a generic telemetry CSV into a sorted list of `Telemetry`.

    Expected header (order-independent): `timestamp, latitude, longitude,
    altitude_msl, yaw, pitch, roll` plus optional `gimbal_yaw, gimbal_pitch,
    gimbal_roll`. `timestamp` must be ISO-8601 UTC (e.g.
    `2026-07-01T15:42:11Z`).

    Args:
        path: Path to the telemetry CSV file.

    Returns:
        Parsed samples, sorted ascending by timestamp.

    Raises:
        VisionModelError: If the file is missing, has no header, or is
            missing a required column.
        TelemetryNotFoundError: If the file has a valid header but zero
            data rows.
    """
    p = Path(path)
    if not p.is_file():
        raise VisionModelError(f"Telemetry CSV not found: {p}")

    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise VisionModelError(f"Telemetry CSV has no header: {p}")

        missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise VisionModelError(
                f"Telemetry CSV {p} is missing required column(s): {sorted(missing)}"
            )

        samples: list[Telemetry] = []
        for line_num, row in enumerate(reader, start=2):  # header is line 1
            try:
                samples.append(
                    Telemetry(
                        timestamp=row["timestamp"],
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        altitude_msl=float(row["altitude_msl"]),
                        yaw=float(row["yaw"]),
                        pitch=float(row["pitch"]),
                        roll=float(row["roll"]),
                        gimbal_yaw=_optional_float(row, "gimbal_yaw"),
                        gimbal_pitch=_optional_float(row, "gimbal_pitch"),
                        gimbal_roll=_optional_float(row, "gimbal_roll"),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise VisionModelError(f"{p}:{line_num}: malformed telemetry row: {exc}") from exc

    if not samples:
        raise TelemetryNotFoundError(f"Telemetry CSV {p} has a valid header but no data rows.")

    samples.sort(key=lambda t: t.timestamp)
    logger.info("Loaded %d telemetry samples from %s", len(samples), p)
    return samples


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


class TelemetryStream:
    """A sorted, time-searchable sequence of `Telemetry` samples.

    Wraps `utils.time_sync.find_nearest` so `geotagger.py` doesn't need to
    know whether telemetry came from a CSV file, an in-memory list, or a
    future ROS 2 topic subscription (Phase 6 extension point) — anything
    that can produce a `list[Telemetry]` can back a `TelemetryStream`.
    """

    def __init__(self, samples: list[Telemetry]) -> None:
        if not samples:
            raise TelemetryNotFoundError("TelemetryStream requires at least one sample.")
        self._samples = sorted(samples, key=lambda t: t.timestamp)

    @classmethod
    def from_csv(cls, path: str | Path) -> TelemetryStream:
        """Build a `TelemetryStream` from a generic telemetry CSV file.

        Args:
            path: Path to a telemetry CSV (see `load_telemetry_csv`).

        Returns:
            A ready-to-query `TelemetryStream`.
        """
        return cls(load_telemetry_csv(path))

    def nearest(self, timestamp: str, max_delta_seconds: float | None = None) -> Telemetry:
        """Look up the telemetry sample nearest `timestamp`.

        Args:
            timestamp: ISO-8601 UTC timestamp to match against.
            max_delta_seconds: If set, raise when the closest sample is
                farther than this many seconds away.

        Returns:
            The nearest `Telemetry` sample.

        Raises:
            TelemetryNotFoundError: If no sample is within
                `max_delta_seconds` (when set).
        """
        return find_nearest(
            timestamp, self._samples, lambda t: t.timestamp, max_delta_seconds=max_delta_seconds
        )

    def __len__(self) -> int:
        return len(self._samples)
