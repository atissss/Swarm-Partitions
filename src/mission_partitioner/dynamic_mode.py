"""
dynamic_mode.py
---------------
Interactive loop that lets the user click polygon vertices on the matplotlib
figure to define dynamic (runtime) no-go zones.

Each round of clicking defines one polygon. The loop ends when the user
presses Enter without clicking at least 3 points.

Bug fixes applied
-----------------
BUG 9  – Dynamic polygons are now validated before being accepted:
           • Self-intersecting or otherwise invalid polygons are repaired or
             rejected with a clear message.
           • Zero-area polygons are rejected.
           • Polygons are clipped to the mission boundary so clicks outside
             the area are silently trimmed rather than creating out-of-bounds
             obstacles.
           • Polygons that are entirely outside the boundary (empty after clip)
             are rejected with a warning.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from .visualiser import draw

# Minimum area (m²) for a dynamic no-go zone to be accepted.
# Prevents accidental single-click polygons from being recorded.
_MIN_AREA_M2 = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_clip(
    poly: Polygon,
    boundary: Polygon,
    index: int,
) -> Polygon | None:
    """
    Validate a click-drawn polygon and clip it to the mission boundary.

    BUG 9 FIX: checks validity, minimum area, and clips to boundary.

    Returns the cleaned polygon, or None if it should be rejected.
    """
    # Repair self-intersections (common from quick clicking)
    if not poly.is_valid:
        reason = explain_validity(poly)
        print(f"[dynamic_mode] Zone {index}: invalid geometry ({reason}) — attempting repair.")
        poly = poly.buffer(0)
        if not poly.is_valid:
            print(f"[dynamic_mode] Zone {index}: repair failed — discarding.")
            return None

    # Clip to boundary
    clipped = poly.intersection(boundary)

    if clipped.is_empty:
        print(f"[dynamic_mode] Zone {index}: entirely outside mission boundary — discarding.")
        return None

    if clipped.area < _MIN_AREA_M2:
        print(
            f"[dynamic_mode] Zone {index}: area {clipped.area:.2f} m² is below "
            f"minimum {_MIN_AREA_M2} m² — discarding."
        )
        return None

    return clipped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_interactive_loop(
    fig: Figure,
    ax: Axes,
    *,
    partitions: list[Polygon],
    predetermined_nogo: list[tuple[str, Polygon]],
    boundary: Polygon,
    n_parts: int,
) -> list[Polygon]:
    """
    Display the partitioned map and let the user draw dynamic no-go zones.

    Usage:
        - Left-click ≥ 3 points to define a polygon obstacle.
        - Press Enter (without clicking) to finish and export.

    Args:
        fig:               Matplotlib Figure.
        ax:                Matplotlib Axes.
        partitions:        Computed flyable partitions.
        predetermined_nogo: Static no-go zones (for redrawing context).
        boundary:          Mission boundary polygon (used for clipping).
        n_parts:           Total partition count (for colour normalisation).

    Returns:
        List of validated, boundary-clipped Shapely Polygons (may be empty).
    """
    dynamic_nogo: list[Polygon] = []

    print("\n--- DYNAMIC MODE ---")
    print("Left-click ≥3 points to add a no-go zone. Press Enter with no clicks to finish.\n")

    while True:
        draw(
            ax,
            partitions=partitions,
            predetermined_nogo=predetermined_nogo,
            dynamic_nogo=dynamic_nogo,
            n_parts=n_parts,
        )

        pts = plt.ginput(n=-1, timeout=0)

        if len(pts) < 3:
            print("[dynamic_mode] Fewer than 3 points — exiting interactive mode.")
            break

        candidate = Polygon(pts)
        zone_index = len(dynamic_nogo) + 1

        # BUG 9 FIX: validate and clip before accepting
        validated = _validate_and_clip(candidate, boundary, zone_index)
        if validated is not None:
            dynamic_nogo.append(validated)
            print(
                f"[dynamic_mode] No-go zone {len(dynamic_nogo)} accepted "
                f"({len(pts)} vertices, area {validated.area:.1f} m²)."
            )

    plt.close(fig)
    return dynamic_nogo