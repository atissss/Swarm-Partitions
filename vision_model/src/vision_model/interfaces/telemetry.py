"""Stable drone telemetry contract, consumed only by `geotagging`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Telemetry:
    """A single drone pose sample.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of this sample.
        latitude: Drone latitude in decimal degrees (WGS84).
        longitude: Drone longitude in decimal degrees (WGS84).
        altitude_msl: Drone altitude in meters, mean sea level.
        yaw: Drone body yaw in degrees, clockwise from true north.
        pitch: Drone body pitch in degrees (nose up positive).
        roll: Drone body roll in degrees (right wing down positive).
        gimbal_yaw: Gimbal yaw offset in degrees, if the camera is on a
            gimbal independent of the body. None implies the camera is
            rigidly mounted and shares the body attitude.
        gimbal_pitch: Gimbal pitch offset in degrees. None implies rigid
            mount.
        gimbal_roll: Gimbal roll offset in degrees. None implies rigid
            mount.

    Raises:
        ValueError: If `latitude`/`longitude` are out of valid WGS84 range.
    """

    timestamp: str
    latitude: float
    longitude: float
    altitude_msl: float
    yaw: float
    pitch: float
    roll: float
    gimbal_yaw: float | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")

    @property
    def effective_yaw(self) -> float:
        """Camera yaw: gimbal yaw if present, otherwise body yaw."""
        return self.gimbal_yaw if self.gimbal_yaw is not None else self.yaw

    @property
    def effective_pitch(self) -> float:
        """Camera pitch: gimbal pitch if present, otherwise body pitch."""
        return self.gimbal_pitch if self.gimbal_pitch is not None else self.pitch

    @property
    def effective_roll(self) -> float:
        """Camera roll: gimbal roll if present, otherwise body roll."""
        return self.gimbal_roll if self.gimbal_roll is not None else self.roll
