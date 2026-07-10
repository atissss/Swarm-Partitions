"""Pixel -> camera-frame ray projection.

Camera frame convention (standard computer-vision / OpenCV convention):
x-right, y-down, z-forward (i.e. z is the optical axis, pointing into the
scene). `frames.py` is responsible for rotating rays out of this frame into
the drone's local NED frame.
"""

from __future__ import annotations

import numpy as np

from vision_model.interfaces.camera import CameraIntrinsics
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)


def pixel_to_ray(pixel: tuple[float, float], intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project an image pixel to a unit ray in the camera frame.

    Args:
        pixel: (u, v) pixel coordinates, e.g. a `Detection.pixel_center`.
        intrinsics: Camera intrinsics used to un-project the pixel.

    Returns:
        A unit-length `np.ndarray` of shape (3,): `[x, y, z]` in the camera
        frame (x-right, y-down, z-forward/into-scene).

    Note:
        Lens distortion correction is not yet applied — `intrinsics.
        distortion` is accepted for forward compatibility but ignored here.
        See the architecture roadmap, Phase 5, for distortion-aware
        back-projection. For narrow-FOV lenses at typical aerial working
        altitudes the resulting geo-tag error from this is usually small,
        but it grows toward image edges on wide-FOV lenses.
    """
    if intrinsics.distortion:
        logger.debug(
            "CameraIntrinsics.distortion is set but not yet applied by pixel_to_ray "
            "(Phase 5 extension point); back-projection assumes an ideal pinhole model."
        )

    u, v = pixel
    fx, fy = intrinsics.focal_length_px
    cx, cy = intrinsics.principal_point_px

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0

    ray = np.array([x, y, z], dtype=np.float64)
    return ray / np.linalg.norm(ray)
