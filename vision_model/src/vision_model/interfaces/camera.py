"""Stable camera model contract, consumed only by `geotagging`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics for a single lens/sensor configuration.

    Attributes:
        focal_length_px: (fx, fy) focal length in pixels.
        principal_point_px: (cx, cy) principal point in pixels.
        image_size_px: (width, height) of the source image in pixels.
        distortion: Optional radial/tangential distortion coefficients in
            OpenCV order (k1, k2, p1, p2, k3, ...). Empty tuple means the
            image is assumed already undistorted (or distortion is
            negligible for the working altitude/lens).
    """

    focal_length_px: tuple[float, float]
    principal_point_px: tuple[float, float]
    image_size_px: tuple[int, int]
    distortion: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        fx, fy = self.focal_length_px
        if fx <= 0 or fy <= 0:
            raise ValueError(f"focal_length_px must be positive, got {self.focal_length_px}")
        w, h = self.image_size_px
        if w <= 0 or h <= 0:
            raise ValueError(f"image_size_px must be positive, got {self.image_size_px}")


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Fixed mount offset of the camera relative to the drone body frame.

    Attributes:
        translation_m: (x, y, z) offset of the camera optical center from
            the drone body origin, in meters, body frame (x-forward,
            y-right, z-down).
        rotation_deg: (yaw, pitch, roll) mount rotation offset in degrees,
            applied on top of the drone's (or gimbal's) attitude.
    """

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
