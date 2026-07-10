"""`python -m vision_model.cli.train` — Hydra-driven training entry point.

Example:
    python -m vision_model.cli.train dataset=visdrone model=yolov_detect \
        train.epochs=100 train.run_name=exp002
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vision_model.cli._common import cli_entrypoint
from vision_model.models.registry import build_detector
from vision_model.training.trainer import Trainer
from vision_model.utils.config_validation import validate_config
from vision_model.utils.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
@cli_entrypoint
def main(cfg: DictConfig) -> None:
    """Compose config, build the detector, and run training.

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

    trainer = Trainer(detector, output_root=cfg.output_root)
    result = trainer.fit(
        data_config=cfg.dataset.yolo_data_yaml,
        epochs=cfg.train.epochs,
        run_name=cfg.train.run_name,
        image_size=cfg.train.image_size,
        seed=cfg.seed,
        resume=cfg.train.get("resume", False),
        batch=cfg.train.batch,
        patience=cfg.train.patience,
    )
    logger.info("Training finished: %s", result["checkpoints"])


if __name__ == "__main__":
    main()
