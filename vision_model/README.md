# vision-model

Aerial object detection (cars, humans) and geo-tagging pipeline for drone imagery — VisDrone / AU-AIR based, built on Ultralytics YOLO.

See [`vision_model_architecture.md`](vision_model_architecture.md) for the full design doc: package responsibilities, data flow, the detector↔geo-tagger interface contract, extension points, and the phased roadmap.

## Implementation status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding, logging, exceptions | ✅ done |
| 1 | MVP detection: datasets, models, inference, training, evaluation, visualization | ✅ done |
| 2 | MVP geo-tagging: telemetry, camera model, projection, frame transforms, exporters | ✅ done |
| 3 | Hardening: config validation, checkpoint resume, CLI error handling, CI | ✅ done |
| 4 | Video & batch inference | ✅ done |
| 5 | Accuracy & robustness (DTM, distortion, multi-camera) | ⬜ not started |
| 6 | Production readiness (live stream, ROS 2, GIS export) | ⬜ not started |

## Install

```bash
pip install -e ".[dev]"
```

Heavy dependencies (`torch`, `ultralytics`, `albumentations`, `opencv-python-headless`) are only imported lazily where actually used — e.g. `vision_model.datasets` and `vision_model.interfaces` work without them installed, which is what keeps the test suite fast and dependency-light.

## Quick start

```python
from vision_model.datasets import build_dataset
from vision_model.models import build_detector
from vision_model.inference import Predictor

# 1. Point at a VisDrone-DET split directory (images/ + annotations/)
dataset = build_dataset("visdrone", root="data/VisDrone2019-DET-val", split="val")
print(f"{len(dataset)} samples")

# 2. Load a detector (requires `pip install ultralytics`)
detector = build_detector("yolo", model_path="runs/exp001/best.pt")
predictor = Predictor(detector, confidence_threshold=0.3)

# 3. Run inference on one image
sample = dataset.index[0]
detections = predictor.predict_image(sample.image_path, timestamp="2026-07-01T15:42:11Z")
for d in detections:
    print(d.class_name, d.confidence, d.bbox)
```

## Geo-tagging quick start

```python
from vision_model.interfaces.camera import CameraIntrinsics, CameraExtrinsics
from vision_model.geotagging.telemetry_loader import TelemetryStream
from vision_model.geotagging.geotagger import tag_detections
from vision_model.geotagging.exporters import export_all

intrinsics = CameraIntrinsics(
    focal_length_px=(1000.0, 1000.0),
    principal_point_px=(960.0, 540.0),
    image_size_px=(1920, 1080),
)
extrinsics = CameraExtrinsics()  # rigid, co-located mount

telemetry = TelemetryStream.from_csv("data/flight_logs/flight1.csv")

# `detections` is whatever `Predictor.predict_image(...)` returned
tagged = tag_detections(
    detections, telemetry, intrinsics, extrinsics,
    ground_elevation_msl=100.0,   # flat-earth assumption; see Phase 5 for DTM support
    skip_errors=True,              # log + skip horizon-grazing detections instead of raising
)

export_all(tagged, output_dir="outputs/geotags", basename="flight1")
# -> outputs/geotags/flight1.{json,csv,geojson}
```

The telemetry CSV needs columns `timestamp, latitude, longitude, altitude_msl, yaw, pitch, roll` (plus optional `gimbal_yaw, gimbal_pitch, gimbal_roll` — gimbal attitude is preferred over body attitude when present). `timestamp` must be ISO-8601 UTC and match (or nearly match) the detections' own `timestamp` field, since that's what's used to time-sync each detection to the nearest telemetry sample.

Run the whole thing (detect + geo-tag + export) via the CLI against a telemetry-enriched dataset (e.g. AU-AIR — VisDrone carries no per-frame timestamps, so it can't be geo-tagged):

```bash
python -m vision_model.cli.geotag dataset=auair \
    model.model_path=runs/exp001/best.pt \
    geotag.telemetry_source=data/flight_logs/flight1.csv \
    camera=dji_mavic3
```

## Video inference quick start

```python
from vision_model.inference.batch_runner import run_on_video
from vision_model.visualization.video_writer import write_annotated_video

# Streams frames one at a time; each frame's timestamp is computed from
# start_timestamp + frame_index / fps, so results plug directly into
# geotagging.tag_detections() for a telemetry-synced flight video.
frames = run_on_video(predictor, "flight1.mp4", start_timestamp="2026-07-01T09:00:00Z")

summary = write_annotated_video(frames, "outputs/flight1_annotated.mp4", fps=30.0)
print(summary.num_frames, summary.num_detections)
```

`run_on_video` and `write_annotated_video` are separate calls specifically so you can insert `geotagging.tag_detections()` between them — detect, geo-tag against a telemetry stream, *then* draw/write — without either package needing to know about the other.

## Training / inference / evaluation via CLI

All five CLI entry points (`train`, `infer`, `infer_video`, `evaluate`, `geotag`) are Hydra entry points — any config value can be overridden on the command line (see `configs/config.yaml` for the full composition).

```bash
# Train (requires `pip install ultralytics torch`)
python -m vision_model.cli.train dataset=visdrone train.epochs=100 train.run_name=exp002

# Resume an interrupted run from its last.pt
python -m vision_model.cli.train train.run_name=exp002 train.resume=true

# Run inference over a configured split, saving annotated images
python -m vision_model.cli.infer model.model_path=runs/exp002/best.pt

# Evaluate precision/recall/F1/mAP against a validation split
python -m vision_model.cli.evaluate model.model_path=runs/exp002/best.pt

# Run inference over a video, writing an annotated copy with boxes/labels drawn
python -m vision_model.cli.infer_video \
    model.model_path=runs/exp002/best.pt \
    inference.video_path=data/flights/flight1.mp4 \
    inference.video_start_timestamp=2026-07-01T09:00:00Z
```

Every CLI entry point validates its composed config up front (`utils.config_validation`) and exits cleanly with a one-line message on a `VisionModelError` instead of a raw traceback — see `cli/_common.py`.

## Tests

```bash
pytest
```

The suite (159 tests as of Phase 4) runs against synthetic fixtures only — no real VisDrone/AU-AIR download, trained weights, or real flight logs required (video tests use a small MJPG/AVI clip generated on the fly). Config-composition and config-validation tests exercise every shipped Hydra config combination (and deliberately-broken variants); geo-tagging tests include hand-computed expected coordinates for known camera/telemetry setups (e.g. "nadir camera + center pixel = directly below the drone"). CI (`.github/workflows/ci.yml`) runs Black, Ruff, and the full suite with coverage on every push.

## Adding a custom dataset

1. Subclass `vision_model.datasets.base.AerialDetectionDataset`, implementing `load_index()` (see `datasets/visdrone.py`).
2. `register("my_dataset", MyDataset)` in `datasets/registry.py`.
3. Copy `configs/dataset/custom_template.yaml` to `configs/dataset/my_dataset.yaml` and fill it in.

No changes to `models`, `training`, `inference`, or `evaluation` are required — see the architecture doc §2/§7 for why.
