from __future__ import annotations

from dataclasses import dataclass

import pytest

from vision_model.utils.exceptions import TelemetryNotFoundError
from vision_model.utils.time_sync import (
    find_nearest,
    format_iso8601,
    frame_timestamp,
    parse_iso8601,
)


@dataclass
class _Sample:
    timestamp: str
    value: int


SAMPLES = [
    _Sample("2026-07-01T15:42:10Z", 1),
    _Sample("2026-07-01T15:42:11Z", 2),
    _Sample("2026-07-01T15:42:13Z", 3),
]


class TestParseIso8601:
    def test_parses_z_suffix(self) -> None:
        dt = parse_iso8601("2026-07-01T15:42:11Z")
        assert dt.hour == 15 and dt.minute == 42 and dt.second == 11

    def test_parses_explicit_offset(self) -> None:
        dt = parse_iso8601("2026-07-01T15:42:11+00:00")
        assert dt.hour == 15


class TestFindNearest:
    def test_exact_match(self) -> None:
        result = find_nearest("2026-07-01T15:42:11Z", SAMPLES, lambda s: s.timestamp)
        assert result.value == 2

    def test_nearest_between_samples(self) -> None:
        # 15:42:12 is 1s from sample[1] (11) and 1s from sample[2] (13);
        # bisect_left ties break toward the earlier candidate deterministically
        result = find_nearest("2026-07-01T15:42:12Z", SAMPLES, lambda s: s.timestamp)
        assert result.value in (2, 3)

    def test_closer_to_later_sample(self) -> None:
        result = find_nearest("2026-07-01T15:42:12.9Z", SAMPLES, lambda s: s.timestamp)
        assert result.value == 3

    def test_empty_samples_raises(self) -> None:
        with pytest.raises(TelemetryNotFoundError):
            find_nearest("2026-07-01T15:42:11Z", [], lambda s: s.timestamp)

    def test_max_delta_exceeded_raises(self) -> None:
        with pytest.raises(TelemetryNotFoundError):
            find_nearest(
                "2026-07-01T16:00:00Z",
                SAMPLES,
                lambda s: s.timestamp,
                max_delta_seconds=5.0,
            )

    def test_max_delta_respected_when_within_bound(self) -> None:
        result = find_nearest(
            "2026-07-01T15:42:11Z",
            SAMPLES,
            lambda s: s.timestamp,
            max_delta_seconds=5.0,
        )
        assert result.value == 2


class TestFormatIso8601:
    def test_round_trips_through_parse(self) -> None:
        original = "2026-07-01T15:42:11Z"
        assert format_iso8601(parse_iso8601(original)) == original

    def test_naive_datetime_assumed_utc(self) -> None:
        from datetime import datetime

        dt = datetime(2026, 7, 1, 15, 42, 11)
        assert format_iso8601(dt) == "2026-07-01T15:42:11Z"


class TestFrameTimestamp:
    def test_frame_zero_equals_start(self) -> None:
        assert frame_timestamp("2026-07-01T00:00:00Z", 0, fps=10.0) == "2026-07-01T00:00:00Z"

    def test_advances_by_frame_period(self) -> None:
        assert frame_timestamp("2026-07-01T00:00:00Z", 1, fps=10.0) == "2026-07-01T00:00:00.100000Z"
        assert frame_timestamp("2026-07-01T00:00:00Z", 5, fps=10.0) == "2026-07-01T00:00:00.500000Z"

    def test_crosses_second_boundary(self) -> None:
        assert frame_timestamp("2026-07-01T00:00:00Z", 30, fps=30.0) == "2026-07-01T00:00:01Z"

    def test_nonpositive_fps_raises(self) -> None:
        with pytest.raises(ValueError):
            frame_timestamp("2026-07-01T00:00:00Z", 0, fps=0.0)
        with pytest.raises(ValueError):
            frame_timestamp("2026-07-01T00:00:00Z", 0, fps=-5.0)

    def test_negative_frame_index_raises(self) -> None:
        with pytest.raises(ValueError):
            frame_timestamp("2026-07-01T00:00:00Z", -1, fps=10.0)
