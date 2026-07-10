"""`python -m vision_model.cli.infer` — Hydra-driven inference entry point.

Example:
    python -m vision_model.cli.infer model=yolov_detect \
        model.model_path=runs/exp001/best.pt \
        inference.output_dir=outputs/my_flight
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from vision_model.cli._common import cli_entrypoint
from vision_model.inference.predictor import Predictor
from vision_model.models.registry import build_detector
from vision_model.utils.config_validation import validate_config
from vision_model.utils.exceptions import VisionModelError
from vision_model.utils.io import ensure_dir
from vision_model.utils.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
@cli_entrypoint
def main(cfg: DictConfig) -> None:
    """Compose config, build the detector, and run inference over a directory.

    Reads images from `cfg.dataset.splits.test.root` (or `val` if no `test`
    split is configured), draws/saves detections if
    `cfg.inference.save_visualizations` is set, and logs a summary.

    Args:
        cfg: Hydra-composed configuration (see `configs/config.yaml`).
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

    split_cfg = cfg.dataset.splits.get("test") or cfg.dataset.splits.val
    images_dir = Path(split_cfg.root) / "images"
    if not images_dir.is_dir():
        raise VisionModelError(f"Images directory not found: {images_dir}")
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    logger.info("Running inference on %d images from %s", len(image_paths), images_dir)

    output_dir = ensure_dir(cfg.inference.output_dir)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    total_detections = 0
    for image_path in image_paths:
        detections = predictor.predict_image(image_path, timestamp=now)
        total_detections += len(detections)

        if cfg.inference.save_visualizations:
            import cv2

            from vision_model.visualization.draw_boxes import save_annotated_image

            image = cv2.imread(str(image_path))
            save_annotated_image(image, detections, output_dir / image_path.name)

    logger.info(
        "Inference complete: %d detections across %d images. Output: %s",
        total_detections,
        len(image_paths),
        output_dir,
    )


if __name__ == "__main__":
    main()
