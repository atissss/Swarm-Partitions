"""`python -m vision_model.cli.evaluate` — Hydra-driven evaluation entry point.

Runs the detector over a configured dataset split and reports
precision/recall/F1/mAP against ground truth (see `evaluation.metrics` for
the standardized, backend-independent computation — this doesn't rely on
Ultralytics' own reported validation metrics).

Example:
    python -m vision_model.cli.evaluate model.model_path=runs/exp001/best.pt \
        dataset=visdrone
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vision_model.cli._common import cli_entrypoint
from vision_model.datasets.registry import build_dataset
from vision_model.evaluation.evaluator import evaluate_dataset
from vision_model.inference.predictor import Predictor
from vision_model.models.registry import build_detector
from vision_model.utils.config_validation import validate_config
from vision_model.utils.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
@cli_entrypoint
def main(cfg: DictConfig) -> None:
    """Compose config, build the detector, and evaluate it against a split.

    Args:
        cfg: Hydra-composed configuration (see `configs/config.yaml`).
            Evaluates against `cfg.dataset.splits.val` by default.
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

    dataset = build_dataset(cfg.dataset.name, root=cfg.dataset.splits.val.root, split="val")
    report = evaluate_dataset(predictor, dataset, class_names=list(cfg.dataset.class_names))

    logger.info("Evaluation results (IoU=%.2f):", report.iou_threshold)
    for class_name, metrics in report.per_class.items():
        logger.info(
            "  %-8s precision=%.3f recall=%.3f f1=%.3f AP=%.3f (gt=%d, pred=%d)",
            class_name,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.average_precision,
            metrics.num_ground_truth,
            metrics.num_predictions,
        )
    logger.info("  mAP@%.2f = %.3f", report.iou_threshold, report.mean_average_precision)


if __name__ == "__main__":
    main()
