"""
mission_partitioner
-------------------
A modular toolkit for partitioning UAV/drone mission areas defined in KML,
generating Voronoi-based sector assignments, and exporting results to JSON.

Public API
----------
    load_kml            – parse a KML file into boundary + no-go polygons
    build_partitions    – run KMeans + Voronoi partitioning
    run_interactive_loop – interactive dynamic no-go zone drawing
    export_json         – serialise mission output to JSON
    draw                – render the mission map onto a matplotlib Axes
"""

from .kml_loader import load_kml
from .partitioner import build_partitions
from .dynamic_mode import run_interactive_loop
from .exporter import export_json
from .visualiser import draw, make_figure

__all__ = [
    "load_kml",
    "build_partitions",
    "run_interactive_loop",
    "export_json",
    "draw",
    "make_figure",
]