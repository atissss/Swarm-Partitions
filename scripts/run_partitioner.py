"""
scripts/run_partitioner.py
--------------------------
Command-line entry point for the mission partitioner.

Usage
-----
    python scripts/run_partitioner.py --kml data/input/Mission_Area.kml --parts 5
    python scripts/run_partitioner.py --kml data/input/Mission_Area.kml --parts 5 \\
        --output data/output/result.json --seed 123 \\
        --min-area 1000 --min-width 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mission_partitioner.dynamic_mode import run_interactive_loop
from mission_partitioner.exporter import export_json
from mission_partitioner.kml_loader import load_kml
from mission_partitioner.partitioner import build_partitions
from mission_partitioner.visualiser import make_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition a KML mission area into N flyable sectors."
    )
    parser.add_argument(
        "--kml", required=True,
        help="Path to the input KML file.",
    )
    parser.add_argument(
        "--parts", type=int, required=True,
        help="Number of partitions to generate.",
    )
    parser.add_argument(
        "--output", default="data/output/mission_output.json",
        help="Destination path for the JSON output.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible partitioning (default: 42).",
    )
    parser.add_argument(
        "--min-area", type=float, default=500.0,
        dest="min_area",
        help="Minimum partition fragment area in m² before it is discarded (default: 500).",
    )
    parser.add_argument(
        "--min-width", type=float, default=10.0,
        dest="min_width",
        help="Minimum partition fragment width in metres before it is discarded (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load KML (now also returns home point)
    print(f"[run] Loading KML: {args.kml}")
    kml_data = load_kml(args.kml)

    boundary           = kml_data["boundary"]
    predetermined_nogo = kml_data["nogo"]
    home               = kml_data["home"]       # (lon, lat) or None
    to_meters          = kml_data["to_meters"]
    to_latlon          = kml_data["to_latlon"]
    epsg_code          = kml_data["epsg_code"]
    nogo_polys         = [p for _, p in predetermined_nogo]

    if home is not None:
        print(f"[run] Home point: lon={home[0]:.6f}, lat={home[1]:.6f}")
        home_point = to_meters.transform(home[0], home[1])
    else:
        print("[run] No home point defined in KML.")
        home_point = None

    # 2. Build partitions
    print(
        f"[run] Building {args.parts} partitions "
        f"(seed={args.seed}, min_area={args.min_area} m², min_width={args.min_width} m)..."
    )
    partitions = build_partitions(
        boundary=boundary,
        nogo_polys=nogo_polys,
        n_parts=args.parts,
        random_state=args.seed,
        min_area_m2=args.min_area,
        min_width_m=args.min_width,
    )
    print(f"[run] {len(partitions)} partitions created.")

    # 3. Interactive dynamic no-go loop
    fig, ax = make_figure()
    dynamic_nogo = run_interactive_loop(
        fig, ax,
        partitions=partitions,
        predetermined_nogo=predetermined_nogo,
        boundary=boundary,
        n_parts=args.parts,
        home_point=home_point,
    )

    # 4. Export JSON
    export_json(
        boundary=boundary,
        partitions=partitions,
        predetermined_nogo=predetermined_nogo,
        dynamic_nogo=dynamic_nogo,
        to_latlon=to_latlon,
        epsg_code=epsg_code,
        home=home,
        random_seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()