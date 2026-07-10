"""Coordinate frame transformations: camera -> body -> NED -> WGS84.

Rotation convention used throughout this module:

- Euler angles (yaw, pitch, roll) follow the standard aerospace 3-2-1
  (Z-Y-X) sequence: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, transforming a
  vector expressed in a body-like frame (x-forward, y-right, z-down) into
  the local NED frame (north, east, down). yaw=0/pitch=0/roll=0 means the
  body's forward axis points true north, horizontally.
- pitch follows the common drone-gimbal convention: 0 = horizontal, -90 =
  straight down, +90 = straight up (this matches `Telemetry.gimbal_pitch`
  usage elsewhere in this codebase, e.g. the nadir fixture in `conftest.py`).
- The camera's own optical frame (x-right, y-down, z-forward-into-scene,
  see `camera_model.py`) is related to this body-like frame by a fixed axis
  permutation (`camera_to_body_axes`), *before* any rotation is applied:
  camera-forward (z) <-> body-forward (x), camera-right (x) <-> body-right
  (y), camera-down (y) <-> body-down (z).
"""

from __future__ import annotations

import numpy as np
import pymap3d as pm

from vision_model.interfaces.camera import CameraExtrinsics
from vision_model.interfaces.telemetry import Telemetry

# Fixed axis permutation: camera frame (x-right, y-down, z-forward) ->
# body-like frame (x-forward, y-right, z-down), at zero mount rotation.
_CAMERA_TO_BODY_AXES = np.array(
    [
        [0.0, 0.0, 1.0],  # body-forward = camera-z
        [1.0, 0.0, 0.0],  # body-right   = camera-x
        [0.0, 1.0, 0.0],  # body-down    = camera-y
    ]
)


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build the body-like-frame -> NED rotation matrix from Euler angles.

    Args:
        yaw_deg: Yaw in degrees, clockwise from true north.
        pitch_deg: Pitch in degrees (0 = horizontal, -90 = straight down).
        roll_deg: Roll in degrees (right-side-down positive).

    Returns:
        A (3, 3) rotation matrix `R` such that `v_ned = R @ v_body`.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    r_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    return r_z @ r_y @ r_x


def camera_ray_to_ned(
    ray_camera: np.ndarray,
    telemetry: Telemetry,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """Rotate a camera-frame ray into the local NED frame.

    Composes: mount rotation (fixed camera-to-body offset) then drone/
    gimbal attitude (body-to-NED), on top of the fixed camera<->body axis
    permutation.

    Args:
        ray_camera: Unit ray in the camera frame (x-right, y-down,
            z-forward), as returned by `camera_model.pixel_to_ray`.
        telemetry: Drone telemetry sample; `effective_yaw/pitch/roll`
            (gimbal if present, else body) supply the attitude rotation.
        extrinsics: Fixed camera mount offset relative to the drone body.

    Returns:
        A unit-length `np.ndarray` of shape (3,): `[north, east, down]`
        ray direction.
    """
    r_mount = euler_to_rotation_matrix(*extrinsics.rotation_deg)
    r_attitude = euler_to_rotation_matrix(
        telemetry.effective_yaw, telemetry.effective_pitch, telemetry.effective_roll
    )

    ray_body = r_mount @ (_CAMERA_TO_BODY_AXES @ ray_camera)
    ray_ned = r_attitude @ ray_body
    return ray_ned / np.linalg.norm(ray_ned)


def ned_offset_to_geodetic(
    ned_offset: tuple[float, float, float],
    ref_latitude: float,
    ref_longitude: float,
    ref_altitude_msl: float,
) -> tuple[float, float, float]:
    """Convert a local NED offset from a reference point to WGS84 geodetic coordinates.

    Args:
        ned_offset: (north, east, down) offset in meters from the reference
            point.
        ref_latitude: Reference point latitude in decimal degrees.
        ref_longitude: Reference point longitude in decimal degrees.
        ref_altitude_msl: Reference point altitude in meters. Treated as
            height above the WGS84 ellipsoid for this transform (a standard
            simplification — the MSL/ellipsoid difference, the geoid
            undulation, is typically tens of meters and is a known source
            of systematic vertical offset; correcting it is a Phase 5
            extension point, e.g. via a geoid model).

    Returns:
        (latitude, longitude, altitude) in decimal degrees / meters.
    """
    n, e, d = ned_offset
    lat, lon, alt = pm.ned2geodetic(n, e, d, ref_latitude, ref_longitude, ref_altitude_msl)
    return float(lat), float(lon), float(alt)
