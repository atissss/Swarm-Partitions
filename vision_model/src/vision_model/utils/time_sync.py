"""Timestamp alignment between detections (video frames) and telemetry samples.

Kept dependency-free (stdlib only) so it's trivially unit-testable and usable
from both `inference` and `geotagging` without pulling in geo libraries.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from vision_model.utils.exceptions import TelemetryNotFoundError

T = TypeVar("T")


def parse_iso8601(timestamp: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string into a timezone-aware datetime.

    Args:
        timestamp: ISO-8601 string, e.g. "2026-07-01T15:42:11Z" or with an
            explicit "+00:00" offset.

    Returns:
        A timezone-aware `datetime` in UTC.
    """
    ts = timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_iso8601(dt: datetime) -> str:
    """Format a timezone-aware `datetime` as an ISO-8601 UTC string.

    Args:
        dt: A `datetime`. Naive datetimes are assumed to already be UTC.

    Returns:
        An ISO-8601 string with a `Z` suffix, e.g. `"2026-07-01T15:42:11.500000Z"`.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def frame_timestamp(start_timestamp: str, frame_index: int, fps: float) -> str:
    """Compute a video frame's capture timestamp from its index and frame rate.

    Args:
        start_timestamp: ISO-8601 UTC timestamp of frame 0 (e.g. when
            recording began).
        frame_index: Zero-based frame index within the video.
        fps: Video frame rate in frames per second. Must be positive.

    Returns:
        ISO-8601 UTC timestamp for `frame_index`, i.e.
        `start_timestamp + frame_index / fps` seconds.

    Raises:
        ValueError: If `fps` is not positive or `frame_index` is negative.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")

    start = parse_iso8601(start_timestamp)
    offset = timedelta(seconds=frame_index / fps)
    return format_iso8601(start + offset)


def find_nearest(  # noqa: UP047 - TypeVar kept for broader tooling/readability compatibility
    target_timestamp: str,
    samples: Sequence[T],
    get_timestamp: callable[[T], str],
    max_delta_seconds: float | None = None,
) -> T:
    """Find the sample in `samples` whose timestamp is closest to `target_timestamp`.

    Args:
        target_timestamp: ISO-8601 timestamp to match against.
        samples: A sequence of samples, assumed sorted by timestamp ascending.
        get_timestamp: Callable extracting an ISO-8601 timestamp string from a
            sample (e.g. `lambda t: t.timestamp`).
        max_delta_seconds: If set, raise if the closest match is farther than
            this many seconds away.

    Returns:
        The nearest sample.

    Raises:
        TelemetryNotFoundError: If `samples` is empty, or if the nearest match
            exceeds `max_delta_seconds`.
    """
    if not samples:
        raise TelemetryNotFoundError("No samples available to match against.")

    target = parse_iso8601(target_timestamp)
    sample_times = [parse_iso8601(get_timestamp(s)) for s in samples]

    idx = bisect_left(sample_times, target)

    candidates = []
    if idx < len(samples):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)

    best_idx = min(candidates, key=lambda i: abs((sample_times[i] - target).total_seconds()))
    best_delta = abs((sample_times[best_idx] - target).total_seconds())

    if max_delta_seconds is not None and best_delta > max_delta_seconds:
        raise TelemetryNotFoundError(
            f"Nearest sample is {best_delta:.3f}s from target {target_timestamp}, "
            f"exceeding max_delta_seconds={max_delta_seconds}."
        )

    return samples[best_idx]
