# Vision-Model Branch — Architecture & Design Document

**Scope:** Aerial object detection (cars, humans) + geo-tagging pipeline
**Datasets:** VisDrone (primary), AU-AIR (telemetry-enriched)
**Stack:** Python 3.12+, PyTorch, Ultralytics YOLO, OpenCV, Albumentations, pyproj, GeographicLib, pymap3d, Hydra/YAML, pytest

This document covers the six pre-implementation deliverables: package layout, module responsibilities, data flow, the detector↔geotagger interface, extension points, and a phased roadmap. No implementation code is included here — this is the design to review before Phase 1 work begins.

---

## 1. Design Principles

- **Loose coupling.** The detector and geo-tagger never import each other's internals. They communicate only through the frozen `Detection` / `Telemetry` / `CameraParams` dataclasses defined in `interfaces/`. Either side can be swapped (e.g. YOLO → RT-DETR, or a new projection model) without touching the other.
- **Config-driven, not hardcoded.** Every tunable (model size, image size, confidence threshold, camera intrinsics, DTM source) lives in YAML, composed via Hydra. No magic numbers in code.
- **Single Responsibility per package.** Each top-level package answers exactly one question (see §3).
- **Pipeline stages are pure functions where possible.** `frame → detections`, `(detections, telemetry) → geo-tagged detections`, `geo-tagged detections → GeoJSON`. This makes unit testing tractable without mocking half the system.
- **Fail loud, fail structured.** Custom exception hierarchy per package; no bare `except`, no silent coordinate NaNs — invalid geo-tags are explicitly flagged (`GeoTagError`, out-of-frustum, ray-parallel-to-ground, etc.) rather than dropped or defaulted.

---

## 2. Repository / Package Structure

```
vision-model/
├── configs/
│   ├── config.yaml                  # Hydra root config
│   ├── dataset/
│   │   ├── visdrone.yaml
│   │   ├── auair.yaml
│   │   └── custom_template.yaml
│   ├── model/
│   │   ├── yolov_detect.yaml
│   ├── camera/
│   │   ├── dji_mavic3.yaml           # per-airframe intrinsics/mount offsets
│   │   └── generic_pinhole.yaml
│   ├── train.yaml
│   ├── inference.yaml
│   └── geotag.yaml
│
├── src/vision_model/
│   ├── __init__.py
│   ├── interfaces/                   # <-- stable contracts, imported everywhere
│   │   ├── detection.py              # Detection dataclass
│   │   ├── telemetry.py              # Telemetry dataclass
│   │   ├── camera.py                 # CameraIntrinsics / CameraExtrinsics
│   │   └── geotag.py                 # GeoTaggedDetection dataclass
│   │
│   ├── datasets/
│   │   ├── base.py                   # AerialDetectionDataset ABC
│   │   ├── visdrone.py
│   │   ├── auair.py
│   │   ├── annotation_parsers.py     # per-format parsers -> unified schema
│   │   ├── augmentations.py          # Albumentations pipelines
│   │   └── registry.py               # name -> Dataset class lookup
│   │
│   ├── models/
│   │   ├── base.py                   # Detector ABC (train/val/predict/export)
│   │   ├── yolo_detector.py          # Ultralytics YOLO wrapper
│   │   ├── checkpointing.py
│   │   └── registry.py
│   │
│   ├── training/
│   │   ├── trainer.py                # orchestrates fit loop via model.base
│   │   ├── seed.py                   # reproducibility
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # mAP, precision, recall, F1
│   │   └── evaluator.py
│   │
│   ├── inference/
│   │   ├── predictor.py              # frame(s) -> list[Detection]
│   │   ├── batch_runner.py           # directory / video -> detections
│   │   └── postprocess.py            # NMS, class filtering, confidence gating
│   │
│   ├── geotagging/
│   │   ├── telemetry_loader.py       # parse flight logs -> Telemetry stream
│   │   ├── camera_model.py           # intrinsics/extrinsics, pixel -> ray
│   │   ├── projection.py             # ray-ground intersection (flat + DTM)
│   │   ├── frames.py                 # camera -> body -> NED/ENU -> ECEF -> WGS84
│   │   ├── geotagger.py              # orchestrator: Detections+Telemetry -> GeoTaggedDetections
│   │   └── exporters.py              # JSON / CSV / GeoJSON writers
│   │
│   ├── visualization/
│   │   ├── draw_boxes.py             # overlay detections on frames
│   │   ├── draw_geotags.py           # plot geo-tags on a map/basemap
│   │   └── video_writer.py
│   │
│   ├── utils/
│   │   ├── logging_setup.py
│   │   ├── io.py                     # path handling, safe read/write
│   │   ├── time_sync.py              # timestamp alignment (frame <-> telemetry)
│   │   └── exceptions.py             # exception hierarchy
│   │
│   └── cli/
│       ├── train.py                  # `python -m vision_model.cli.train`
│       ├── infer.py
│       └── geotag.py
│
├── tests/
│   ├── datasets/
│   ├── models/
│   ├── inference/
│   ├── geotagging/
│   │   ├── test_camera_model.py
│   │   ├── test_projection.py
│   │   ├── test_frames.py
│   │   └── test_geotagger.py
│   ├── interfaces/
│   └── conftest.py                   # shared fixtures (synthetic telemetry, cameras)
│
├── pyproject.toml                    # black/ruff config, deps
└── README.md
```

Every leaf package exposes a small `__init__.py` re-exporting its public API, so downstream consumers (tracking, GIS, mission planning) import from `vision_model.geotagging` or `vision_model.inference`, never from internal submodules.

---

## 3. Module Responsibilities

| Package | Responsibility | Explicitly NOT responsible for |
|---|---|---|
| `interfaces` | Define the frozen data contracts (`Detection`, `Telemetry`, `CameraIntrinsics/Extrinsics`, `GeoTaggedDetection`) that cross module boundaries. | Any computation. |
| `datasets` | Load raw imagery + annotations for VisDrone/AU-AIR/custom sources into one unified in-memory schema; apply augmentations. | Knowing anything about geo-tagging or camera models. |
| `models` | Wrap the detector (Ultralytics YOLO) behind a stable `Detector` ABC: `train()`, `validate()`, `predict()`, `export()`, checkpoint save/load. | Dataset parsing, geo-math, visualization. |
| `training` | Orchestrate the fit loop, seeding, callbacks (early stopping, LR schedules), delegate actual optimization to `models`. | Metric definitions (delegates to `evaluation`). |
| `evaluation` | Compute mAP/precision/recall/F1 against ground truth; produce evaluation reports. | Training loop control. |
| `inference` | Run a trained detector on frames/video/directories, post-process (NMS, thresholds), emit `Detection` objects with **pixel coordinates only** — no geography. | Any lat/lon computation. |
| `geotagging` | Consume `Detection` + `Telemetry` + `CameraIntrinsics/Extrinsics`, project pixel → ray → ground → WGS84, emit `GeoTaggedDetection`; export JSON/CSV/GeoJSON. | Running the detector; touching raw images. |
| `visualization` | Draw boxes/labels on frames, plot geo-tags on a basemap, write annotated video. | Business logic — purely presentational. |
| `utils` | Logging setup, IO helpers, timestamp alignment between video frames and telemetry samples, shared exception types. | Domain logic. |
| `configs` | Hydra/YAML composition root: dataset, model, camera, training, inference, geotag configs. | Code. |
| `cli` | Thin entry points wiring config → orchestrator calls (train/infer/geotag). | Business logic (delegates immediately). |
| `tests` | Unit/integration coverage for every package above, with synthetic fixtures for geo-math so tests don't depend on real flight logs. | — |

**Key boundary:** `inference` produces `Detection` objects with `pixel_center`, bbox, class, confidence — and *no* lat/lon fields populated. `geotagging` is the *only* module allowed to populate `latitude`/`longitude`/`altitude`. This is what keeps the two subsystems independently testable and replaceable.

---

## 4. Data Flow Through the Pipeline

```
┌─────────────────┐     ┌──────────────────┐      ┌───────────────────┐
│  Raw drone       │     │  Flight log /    │      │  Camera config     │
│  frame(s) /      │     │  telemetry (GPS, │      │  (intrinsics,      │
│  video stream     │     │  alt, yaw/pitch/ │      │  mount extrinsics) │
│                  │     │  roll, timestamp) │      │                    │
└────────┬─────────┘     └────────┬─────────┘      └─────────┬──────────┘
         │                        │                           │
         ▼                        │                           │
┌─────────────────┐               │                           │
│ inference        │               │                           │
│ .predictor       │               │                           │
│  frame -> [Detection]           │                           │
│  (bbox, class,   │               │                           │
│   confidence,    │               │                           │
│   pixel_center,  │               │                           │
│   timestamp)     │               │                           │
└────────┬─────────┘               │                           │
         │                         │                           │
         └────────────┬────────────┴─────────────┬─────────────┘
                       ▼                          ▼
              ┌─────────────────────────────────────────┐
              │  geotagging.geotagger                     │
              │  1. time_sync: match Detection.timestamp  │
              │     to nearest Telemetry sample            │
              │  2. camera_model: pixel -> ray in camera   │
              │     frame using intrinsics                 │
              │  3. frames: camera -> body -> NED/ENU      │
              │     using extrinsics + gimbal attitude      │
              │  4. projection: ray-ground intersection     │
              │     (flat-earth or DTM-aware)                │
              │  5. frames: NED/ENU -> ECEF -> WGS84 (lat/lon)│
              │     via pymap3d / pyproj / GeographicLib      │
              └────────────────────┬──────────────────────┘
                                   ▼
                        ┌───────────────────────┐
                        │  GeoTaggedDetection     │
                        │  (Detection fields +     │
                        │   lat, lon, alt)          │
                        └──────────┬────────────────┘
                                   ▼
                 ┌─────────────────────────────────────┐
                 │  geotagging.exporters                 │
                 │  -> JSON / CSV / GeoJSON               │
                 └──────────┬──────────────┬─────────────┘
                             ▼              ▼
                  ┌─────────────────┐  ┌──────────────────────┐
                  │ visualization     │  │ Downstream consumers  │
                  │ (overlay/plot)     │  │ (tracking, GIS,        │
                  │                   │  │  mission planning)      │
                  └───────────────────┘  └────────────────────────┘
```

Two upstream tracks (imagery → detections, telemetry+camera-config → pose) merge only inside `geotagger`, which is the single point of coupling by design.

---

## 5. Interfaces Between Detection and Geo-tagging

These four dataclasses are the entire public contract. They live in `interfaces/` and are versioned independently of internal module code — changing their shape is a breaking change requiring a version bump.

```python
# interfaces/detection.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Detection:
    """Raw detector output in pixel/image space. No geography here."""
    class_name: Literal["car", "person"]
    confidence: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    pixel_center: tuple[float, float]          # cx, cy in pixels
    frame_id: str                              # source frame/video identifier
    timestamp: str                             # ISO-8601 UTC, matched against telemetry


# interfaces/telemetry.py
@dataclass(frozen=True)
class Telemetry:
    """One drone pose sample."""
    timestamp: str          # ISO-8601 UTC
    latitude: float
    longitude: float
    altitude_msl: float      # meters, mean sea level
    yaw: float                # degrees
    pitch: float               # degrees
    roll: float                 # degrees
    gimbal_yaw: float | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None


# interfaces/camera.py
@dataclass(frozen=True)
class CameraIntrinsics:
    focal_length_px: tuple[float, float]   # fx, fy
    principal_point_px: tuple[float, float] # cx, cy
    image_size_px: tuple[int, int]           # width, height
    distortion: tuple[float, ...] = ()

@dataclass(frozen=True)
class CameraExtrinsics:
    """Fixed mount offset of camera relative to drone body frame."""
    translation_m: tuple[float, float, float]   # x, y, z offset from body origin
    rotation_deg: tuple[float, float, float]     # mount yaw, pitch, roll offset


# interfaces/geotag.py
@dataclass(frozen=True)
class GeoTaggedDetection:
    """Final stable output — matches the spec in the project brief."""
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    pixel_center: tuple[float, float]
    latitude: float
    longitude: float
    altitude: float
    timestamp: str
```

**Contract rules:**
1. `inference` depends only on `interfaces.detection`. It has no import of `interfaces.telemetry`/`camera`/`geotag`.
2. `geotagging` depends on all four but never imports from `models`, `datasets`, or `inference` implementation modules — only the `Detection` type.
3. `geotagger.tag(detections: list[Detection], telemetry: TelemetrySource, camera: CameraIntrinsics, extrinsics: CameraExtrinsics) -> list[GeoTaggedDetection]` is the single public entry point downstream code should call.
4. Anything that fails to geo-tag (e.g. detection near horizon where the ray never meets the ground, missing telemetry for the timestamp) raises a typed `GeoTagError` rather than emitting a garbage coordinate — callers decide whether to skip, retry, or halt.

---

## 6. Geo-tagging Math (module: `geotagging`)

1. **`telemetry_loader`** — parses flight logs (DJI SRT/CSV, AU-AIR telemetry, PX4/ArduPilot logs) into a sorted `Telemetry` stream; `time_sync` finds the nearest (or interpolated) sample for a given detection timestamp.
2. **`camera_model`** — un-distorts the pixel, converts `pixel_center` → a normalized camera-frame ray using intrinsics (`fx, fy, cx, cy`).
3. **`frames`** — rotates the ray from camera frame → body frame (via `CameraExtrinsics` mount offset) → local NED/ENU frame (via drone yaw/pitch/roll and gimbal attitude from `Telemetry`).
4. **`projection`** — intersects the ray with the ground plane. Two modes:
   - *Flat-earth assumption*: ground at `altitude_msl - altitude_agl = 0`, closed-form intersection.
   - *DTM-aware*: iterative ray-marching against a digital terrain model (extension point for uneven terrain — see §7).
5. **`frames`** (second pass) — converts the local NED/ENU ground point back through ECEF to WGS84 lat/lon using `pymap3d` (NED↔ECEF↔geodetic) cross-checked with `pyproj`/`GeographicLib` for geodesic accuracy over longer ranges.
6. **`geotagger`** — orchestrates steps 1–5 per detection and assembles `GeoTaggedDetection`.

This is intentionally decomposed so each stage (`camera_model`, `frames`, `projection`) is independently unit-testable with synthetic inputs — e.g. "camera pointed straight down from 100m at zero roll/pitch should place the ground point directly below the drone" is a deterministic test case requiring no real data.

---

## 7. Extension Points

The architecture is deliberately seamed at these points for future work:

- **Object tracking** — consumes a time-ordered stream of `GeoTaggedDetection` (or pre-geotag `Detection`) per frame; slots in as a new `tracking/` package sitting between `inference` and `geotagging`, or after `geotagging` for geo-space tracking. No changes needed to either existing module — it only needs the stable dataclasses.
- **Live video inference** — `inference.predictor` already operates per-frame; a `LiveStreamRunner` in `inference/` can wrap an RTSP/UDP feed and call the same `predict(frame)` used for batch/video, publishing `Detection`s onto a queue that `geotagger` consumes in a streaming fashion.
- **ROS 2 integration** — a future `ros2_bridge/` package would subscribe to camera/telemetry topics, construct `Telemetry`/frame inputs, call the existing `inference`/`geotagging` APIs, and republish `GeoTaggedDetection` as a custom ROS message — no core module changes.
- **Multi-camera drones** — `CameraIntrinsics`/`CameraExtrinsics` are already per-camera; `geotagger.tag()` accepts a camera argument per call, so a multi-camera rig just calls it once per camera stream and merges results downstream. A `camera/registry.yaml` keyed by camera ID supports this in config.
- **GIS / mission-planning integration** — `exporters.py` already emits GeoJSON, the lingua franca for QGIS/ArcGIS/PostGIS ingestion; a future `gis/` package can add direct PostGIS writes or WMS/WMTS tile overlays without touching detection or geo-math code.
- **Custom datasets** — `datasets.base.AerialDetectionDataset` is an ABC; new sources implement `load_annotations()` and `__getitem__()` and register themselves in `datasets/registry.py`. No changes to `models` or `training`.

---

## 8. Configuration Strategy

Hydra composes a root `config.yaml` from `dataset/`, `model/`, `camera/`, plus task-level `train.yaml` / `inference.yaml` / `geotag.yaml`. Example composition for a training run:

```
python -m vision_model.cli.train dataset=visdrone model=yolov_detect train.epochs=100
```

Camera configs (`configs/camera/*.yaml`) store per-airframe intrinsics and mount extrinsics so the same code runs across different drone/camera combinations by swapping one config file — this is what keeps `geotagging` decoupled from any specific hardware.

---

## 9. Testing Strategy

| Area | Approach |
|---|---|
| Dataset loading / annotation parsing | Small synthetic fixtures per format (VisDrone/AU-AIR) checked into `tests/datasets/fixtures/`; assert unified schema output. |
| Detection pipeline | Mock detector returning fixed boxes; assert post-processing (NMS, thresholding) behavior in isolation from real model weights. |
| Camera model / projection / frame transforms | Pure-math unit tests with synthetic telemetry (e.g. nadir-pointing camera at known altitude → known ground point), independent of any image or trained model. |
| Coordinate transformations | Cross-check `pymap3d` output against `pyproj`/`GeographicLib` for a battery of known lat/lon/alt triples; assert round-trip (WGS84→ECEF→WGS84) error below a tight epsilon. |
| Geotagger orchestration | End-to-end with synthetic `Detection` + `Telemetry` + camera config → assert output `GeoTaggedDetection` matches hand-computed expected coordinates. |
| Output object validation | Schema/type tests on all four `interfaces` dataclasses, including frozen/immutability guarantees. |
| Configuration loading | Hydra composition tests ensuring every shipped config combination instantiates without error. |

---

## 10. Phased Implementation Roadmap

**Phase 0 — Scaffolding**
Repo skeleton, `pyproject.toml` (black/ruff config), logging setup, exception hierarchy, CI stub, empty package `__init__.py`s with docstrings.

**Phase 1 — MVP Detection**
`datasets.visdrone` + `annotation_parsers`, `models.yolo_detector` wrapping Ultralytics YOLO, basic `training.trainer`, `inference.predictor` on static images, `visualization.draw_boxes`. Goal: train and run inference on VisDrone, produce `Detection` objects, no geo-tagging yet.

**Phase 2 — MVP Geo-tagging**
`interfaces.telemetry`/`camera`/`geotag`, `geotagging.camera_model` + `frames` + `projection` (flat-earth only), `geotagger` orchestrator, `exporters` (JSON/CSV/GeoJSON). Validate against AU-AIR's telemetry-annotated frames since it ships synchronized GPS/attitude data. Goal: end-to-end `frame + telemetry → GeoTaggedDetection`.

**Phase 3 — Evaluation & Hardening**
`evaluation.metrics` (mAP/precision/recall/F1), full test suite from §9, checkpointing/resume, config validation, error handling pass, logging pass, docstring/type-hint completion, Ruff/Black CI gate.

**Phase 4 — Video & Batch Inference**
`inference.batch_runner` for video/directory input, timestamp-based `time_sync` between video frames and telemetry streams, `visualization.video_writer` for annotated + geo-tagged output video.

**Phase 5 — Accuracy & Robustness**
DTM-aware ground intersection in `projection` (replacing flat-earth assumption for hilly terrain), distortion-aware `camera_model`, multi-camera config support, augmentation tuning, dataset expansion (custom drone data via the `datasets` extension point).

**Phase 6 — Production Readiness / Extension Hooks**
Live-stream inference runner, ROS 2 bridge package, GIS export hardening (PostGIS/WMS), tracking module integration point, packaging for deployment (containerization, model export/quantization for edge inference).

Each phase ends with its own passing test suite and is mergeable to `vision-model` independently — later phases don't require earlier ones to be "finished," only their public interfaces to be stable.

---

## 11. Next Steps

This document defines the contracts and structure; no implementation exists yet. Once you confirm this architecture (or flag changes — e.g. different dataset priorities, a different projection strategy, or a different phase ordering), I'll start Phase 0/1: scaffolding the repo and implementing the VisDrone dataset loader + YOLO detector wrapper as the first testable slice.
