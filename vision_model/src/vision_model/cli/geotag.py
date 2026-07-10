"""`python -m vision_model.cli.geotag` — Hydra-driven detect + geo-tag entry point.

Runs the detector over a configured dataset split, geo-tags each frame's
detections against a telemetry CSV (time-synced via each sample's own
`telemetry_timestamp`), and exports the combined result as JSON/CSV/GeoJSON.
This is the full pipeline end-to-end: `inference` and `geotagging` are wired
together here, but neither package imports the other directly (see
`vision_model_architecture.md` §5).

Note: only telemetry-enriched dataset sources populate
`ImageSample.telemetry_timestamp` (e.g. `datasets.auair` — see its
docstring). Plain VisDrone samples carry no capture timestamp and are
skipped here with a warning, since there's nothing to time-sync against; if
your source doesn't ship per-frame timestamps, geo-tagging isn't possible
without another way to associate each frame with a telemetry sample (e.g.
video frame index against a known FPS and start time).

Example:
    python -m vision_model.cli.geotag dataset=auair \
        model.model_path=runs/exp001/best.pt \
        geotag.telemetry_source=data/flight_logs/flight1.csv \
        camera=dji_mavic3
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vision_model.cli._common import cli_entrypoint
from vision_model.datasets.registry import build_dataset
from vision_model.geotagging.exporters import export_all
from vision_model.geotagging.geotagger import tag_detections
from vision_model.geotagging.telemetry_loader import TelemetryStream
from vision_model.inference.predictor import Predictor
from vision_model.interfaces.camera import CameraExtrinsics, CameraIntrinsics
from vision_model.interfaces.detection import Detection
from vision_model.models.registry import build_detector
from vision_model.utils.config_validation import validate_config
from vision_model.utils.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
@cli_entrypoint
def main(cfg: DictConfig) -> None:
    """Compose config, run detection + geo-tagging, and export the results.

    Args:
        cfg: Hydra-composed configuration (see `configs/config.yaml`).
            Images/annotations come from `cfg.dataset` (must be a
            telemetry-enriched source), telemetry from
            `cfg.geotag.telemetry_source`, camera parameters from
            `cfg.camera`.
    """
    configure_logging()
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    validate_config(cfg)

    detector = build_detector(
        cfg.model.name,
        model_path=cfg.model.model_path,
        class_map={int(k): v for k, v in cfg.model.class_map.items()},
        device=cfg.model.device,
    )
    predictor = Predictor(
        detector,
        confidence_threshold=cfg.inference.confidence_threshold,
        iou_threshold=cfg.inference.iou_threshold,
        allowed_classes=set(cfg.inference.allowed_classes),
    )

    intrinsics = CameraIntrinsics(
        focal_length_px=tuple(cfg.camera.intrinsics.focal_length_px),
        principal_point_px=tuple(cfg.camera.intrinsics.principal_point_px),
        image_size_px=tuple(cfg.camera.intrinsics.image_size_px),
        distortion=tuple(cfg.camera.intrinsics.distortion),
    )
    extrinsics = CameraExtrinsics(
        translation_m=tuple(cfg.camera.extrinsics.translation_m),
        rotation_deg=tuple(cfg.camera.extrinsics.rotation_deg),
    )
    telemetry_stream = TelemetryStream.from_csv(cfg.geotag.telemetry_source)

    split_cfg = cfg.dataset.splits.get("test") or cfg.dataset.splits.val
    dataset = build_dataset(cfg.dataset.name, root=split_cfg.root, split="geotag")
    logger.info("Running detect+geotag on %d samples from %s", len(dataset), split_cfg.root)

    all_detections: list[Detection] = []
    skipped_no_timestamp = 0
    for sample in dataset.index:
        if sample.telemetry_timestamp is None:
            skipped_no_timestamp += 1
            continue
        all_detections.extend(
            predictor.predict_image(
                sample.image_path, timestamp=sample.telemetry_timestamp, frame_id=sample.image_id
            )
        )

    if skipped_no_timestamp:
        logger.warning(
            "Skipped %d samples with no telemetry_timestamp (dataset source doesn't "
            "carry per-frame capture time — see this module's docstring).",
            skipped_no_timestamp,
        )

    tagged = tag_detections(
        all_detections,
        telemetry_stream,
        intrinsics,
        extrinsics,
        ground_elevation_msl=cfg.geotag.get("ground_elevation_msl", 0.0),
        max_telemetry_delta_seconds=cfg.geotag.max_telemetry_delta_seconds,
        skip_errors=True,
    )

    output_paths = export_all(tagged, cfg.geotag.output_dir, basename="detections")
    logger.info(
        "Geo-tagging complete: %d tagged detections. Exports: %s", len(tagged), output_paths
    )


if __name__ == "__main__":
    main()
