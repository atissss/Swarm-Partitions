"""
visualiser.py
-------------
All matplotlib rendering for the mission partitioner.
Kept separate so the geometry modules have zero matplotlib dependency
and can be tested headlessly.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from shapely.geometry import MultiPolygon, Polygon


def make_figure() -> tuple[Figure, Axes]:
    """Create and return a (fig, ax) pair sized for mission display."""
    fig, ax = plt.subplots(figsize=(12, 10))
    return fig, ax


def draw(
    ax: Axes,
    *,
    partitions: list[Polygon | MultiPolygon],
    predetermined_nogo: list[tuple[str, Polygon]],
    dynamic_nogo: list[Polygon],
    n_parts: int,
    home_point: tuple[float, float] | None = None,
) -> None:
    """
    Redraw the mission map onto the provided Axes object.

    Colour scheme:
        - Partitions      : pastel Set3 palette, labelled by number
        - Static no-go    : dark red fill, red border
        - Dynamic no-go   : bright red fill with crosshatch
        - Home point      : blue marker with label

    Args:
        ax:                Matplotlib Axes to draw on (cleared on entry).
        partitions:        Flyable partition polygons.
        predetermined_nogo: Static (KML-sourced) no-go zones.
        dynamic_nogo:      Interactively drawn no-go zones.
        n_parts:           Total partition count (used for colour normalisation).
        home_point:        Optional home coordinate in UTM metres.
    """
    ax.clear()
    colors = plt.get_cmap("Set3")

    # --- Partitions ---
    for i, part in enumerate(partitions):
        color = colors(i / max(n_parts, 1))
        geoms = part.geoms if isinstance(part, MultiPolygon) else [part]

        for geom in geoms:
            ax.fill(*geom.exterior.xy, color=color, alpha=0.55, zorder=1)
            ax.plot(*geom.exterior.xy, color="black", linewidth=0.8, zorder=2)

        label_pt = part.representative_point()
        ax.text(
            label_pt.x, label_pt.y,
            str(i + 1),
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            zorder=6,
        )

    # --- Static no-go zones ---
    for _label, ng in predetermined_nogo:
        ax.fill(*ng.exterior.xy, color="darkred", alpha=0.65, zorder=4)
        ax.plot(*ng.exterior.xy, color="red", linewidth=1.5, zorder=4)

    # --- Dynamic no-go zones ---
    for dng in dynamic_nogo:
        ax.fill(*dng.exterior.xy, color="red", alpha=0.85, zorder=5, hatch="xx")

    # --- Home point ---
    if home_point is not None:
        ax.scatter([home_point[0]], [home_point[1]], color="blue", s=80, zorder=7)
        ax.text(
            home_point[0], home_point[1],
            "Home",
            color="blue",
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
            zorder=8,
        )

    ax.set_aspect("equal")
    plt.title("Mission Area — Partitions & No-Go Zones  (click to add obstacles, Enter to finish)")
    plt.draw()