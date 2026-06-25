"""
scripts/run_partitioner.py
--------------------------
Command-line entry point for the mission partitioner.

Usage
-----
    python scripts/run_partitioner.py --kml data/input/Mission_Area.kml --parts 5
    python scripts/run_partitioner.py --kml data/input/Mission_Area.kml --parts 5 --output data/output/result.json --seed 123
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ..src.dynamic_mode import run_interactive_loop
from ..src.exporter import export_json
from ..src.kml_loader import load_kml
from ..src.partitioner import build_partitions
from ..src.visualiser import make_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition a KML mission area into N flyable sectors."
    )
    parser.add_argument(
        "--kml", required=True,
        help="Path to the input KML file (must contain a 'Boundary' placemark).",
    )
    parser.add_argument(
        "--parts", type=int, required=True,
        help="Number of partitions to generate.",
    )
    parser.add_argument(
        "--output", default="data/output/mission_output.json",
        help="Destination path for the JSON output (default: data/output/mission_output.json).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible partitioning (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load KML
    print(f"[run] Loading KML: {args.kml}")
    kml_data = load_kml(args.kml)

    boundary           = kml_data["boundary"]
    predetermined_nogo = kml_data["nogo"]
    to_latlon          = kml_data["to_latlon"]
    epsg_code          = kml_data["epsg_code"]
    nogo_polys         = [p for _, p in predetermined_nogo]

    # 2. Build partitions
    print(f"[run] Building {args.parts} partitions (seed={args.seed})...")
    partitions = build_partitions(
        boundary=boundary,
        nogo_polys=nogo_polys,
        n_parts=args.parts,
        random_state=args.seed,
    )
    print(f"[run] {len(partitions)} partitions created.")

    # 3. Interactive dynamic no-go loop
    fig, ax = make_figure()
    dynamic_nogo = run_interactive_loop(
        fig, ax,
        partitions=partitions,
        predetermined_nogo=predetermined_nogo,
        boundary=boundary,           # passed for BUG 9 clipping fix
        n_parts=args.parts,
    )

    # 4. Export JSON
    export_json(
        boundary=boundary,
        partitions=partitions,
        predetermined_nogo=predetermined_nogo,
        dynamic_nogo=dynamic_nogo,
        to_latlon=to_latlon,
        epsg_code=epsg_code,
        random_seed=args.seed,       # passed for BUG 4 metadata fix
        output_path=args.output,
    )


if __name__ == "__main__":
    main()