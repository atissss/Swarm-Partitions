# Mission Partitioner

A modular Python toolkit for partitioning UAV/drone mission areas defined in KML into **N flyable sectors**, using KMeans clustering and Voronoi diagrams. Supports static (KML-defined) and dynamic (click-to-draw) no-go zones, with JSON export.

---

## Features

- Auto-detects UTM projection from KML coordinates
- KMeans + Voronoi partitioning of flyable workspace
- Handles multi-polygon "island" sectors created by no-go zones
- Interactive matplotlib UI for drawing dynamic obstacles at runtime
- Clean JSON output with lat/lon coordinates

---

## Project Structure

```
mission-partitioner/
├── src/
│   └── mission_partitioner/
│       ├── kml_loader.py       # KML parsing + UTM projection
│       ├── partitioner.py      # KMeans + Voronoi + orphan merging
│       ├── exporter.py         # JSON serialisation
│       ├── visualiser.py       # matplotlib rendering
│       └── dynamic_mode.py     # Interactive no-go zone loop
├── scripts/
│   └── run_partitioner.py      # CLI entry point
├── tests/
│   ├── test_kml_loader.py
│   ├── test_partitioner.py
│   └── test_exporter.py
├── data/
│   ├── input/                  # Place your .kml files here
│   └── output/                 # JSON outputs written here
├── pyproject.toml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mission-partitioner.git
cd mission-partitioner

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python scripts/run_partitioner.py \
    --kml data/input/Mission_Area.kml \
    --parts 5

# Custom output path
python scripts/run_partitioner.py \
    --kml data/input/Mission_Area.kml \
    --parts 5 \
    --output data/output/result.json
```

### KML Requirements

Your KML file must contain:
- One `<Placemark>` whose `<name>` contains the word **`Boundary`** — this defines the mission area.
- Any number of additional `<Placemark>` polygons — treated as **static no-go zones**.

### Interactive Mode

Once the partitions are computed, a matplotlib window opens:
- **Left-click** ≥ 3 points to define a dynamic obstacle polygon.
- **Press Enter** without clicking to finish and write the JSON output.

---

## Output JSON Format

```json
{
    "metadata": { "epsg": "epsg:32643", "n_partitions": 5 },
    "boundary": [[lon, lat], ...],
    "partitions": [
        { "id": 1, "coords": [[lon, lat], ...] },
        ...
    ],
    "no_go_zones": {
        "predetermined": [{ "name": "Lake", "coords": [...] }],
        "dynamic":       [{ "id": 1,       "coords": [...] }]
    }
}
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest
```

---