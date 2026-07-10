"""Ray-ground intersection.

MVP implementation uses a flat-earth assumption: the ground is a horizontal
plane at a configurable elevation. DTM-aware (uneven terrain) intersection
is a Phase 5 extension point — see the architecture roadmap — and would
replace `intersect_flat_ground` with an iterative ray-march against a
terrain model behind the same call signature (ray in, NED offset out), so
`geotagger.py` would need no changes.
"""

from __future__ import annotations

import numpy as np

from vision_model.utils.exceptions import ProjectionError

# Rays whose "down" component is at or below this are treated as parallel to
# (or pointing away from) the ground -- typically a detection near the image
# horizon, or an implausible camera attitude.
_MIN_DOWN_COMPONENT = 1e-6


def intersect_flat_ground(
    ray_ned: np.ndarray,
    height_above_ground_m: float,
) -> tuple[float, float, float]:
    """Intersect a NED-frame ray from the camera with a flat ground plane.

    The camera position is treated as the local NED origin `(0, 0, 0)`; the
    ground plane sits `height_above_ground_m` below it, i.e. at
    `down = height_above_ground_m`.

    Args:
        ray_ned: Unit ray direction in the NED frame, `[north, east, down]`,
            as returned by `frames.camera_ray_to_ned`.
        height_above_ground_m: Camera height above the (flat) ground plane,
            in meters. Must be positive for a physically meaningful result.

    Returns:
        (north, east, down) offset in meters from the camera position to
        the ground intersection point.

    Raises:
        ProjectionError: If the ray does not point downward into the ground
            (e.g. a detection near the horizon, or `ray_ned`'s down
            component is at or below zero), or if `height_above_ground_m`
            is not positive.
    """
    if height_above_ground_m <= 0:
        raise ProjectionError(
            f"height_above_ground_m must be positive, got {height_above_ground_m}. "
            "Check telemetry.altitude_msl vs the configured ground elevation."
        )

    down_component = float(ray_ned[2])
    if down_component <= _MIN_DOWN_COMPONENT:
        raise ProjectionError(
            f"Ray does not intersect the ground plane (down component={down_component:.6f} "
            f"<= {_MIN_DOWN_COMPONENT}); likely a detection near the horizon or an "
            "implausible camera attitude."
        )

    t = height_above_ground_m / down_component
    north, east, down = (t * ray_ned[0], t * ray_ned[1], t * ray_ned[2])
    return float(north), float(east), float(down)
