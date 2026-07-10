"""`python -m vision_model.cli.infer_video` — Hydra-driven video inference entry point.

Runs the detector over every frame of a video and writes an annotated copy
with boxes/labels drawn — see `inference.batch_runner.run_on_video` and
`visualization.video_writer.write_annotated_video`, which this CLI just
wires together with config.

Example:
    python -m vision_model.cli.infer_video \
        model.model_path=runs/exp001/best.pt \
        inference.video_path=data/flights/flight1.mp4 \
        inference.video_start_timestamp=2026-07-01T09:00:00Z
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vision_model.cli._common import cli_entrypoint
from vision_model.inference.batch_runner import get_video_fps, run_on_video
from vision_model.inference.predictor import Predictor
from vision_model.models.registry import build_detector
from vision_model.utils.config_validation import validate_config
from vision_model.utils.logging_setup import configure_logging, get_logger
from vision_model.visualization.video_writer import write_annotated_video

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
@cli_entrypoint
def main(cfg: DictConfig) -> None:
    """Compose config, run detection over a video, and write an annotated copy.

    Args:
        cfg: Hydra-composed configuration (see `configs/config.yaml`).
            Video-specific fields live under `cfg.inference.video_*`.
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

    configured_fps = cfg.inference.get("video_fps")
    output_fps = configured_fps if configured_fps else get_video_fps(cfg.inference.video_path)

    frames = run_on_video(
        predictor,
        cfg.inference.video_path,
        start_timestamp=cfg.inference.video_start_timestamp,
        fps=configured_fps,
    )

    summary = write_annotated_video(
        frames,
        cfg.inference.video_output_path,
        fps=output_fps,
        fourcc=cfg.inference.video_fourcc,
    )

    logger.info(
        "Video inference complete: %d frames, %d detections -> %s",
        summary.num_frames,
        summary.num_detections,
        summary.output_path,
    )


if __name__ == "__main__":
    main()
