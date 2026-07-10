"""Training orchestration.

`Trainer` seeds RNGs, resolves checkpoint paths, and delegates the actual
optimization loop to the `Detector` backend's own `train()` — it does not
reimplement a training loop itself, since Ultralytics YOLO already owns
that. This keeps `Trainer` thin and backend-agnostic: swapping detectors
only requires the new backend to implement `Detector.train()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_model.models.base import Detector
from vision_model.models.checkpointing import CheckpointPaths, resolve_checkpoint_paths
from vision_model.training.seed import set_seed
from vision_model.utils.exceptions import CheckpointError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)


class Trainer:
    """Thin orchestration layer around a `Detector`'s training loop.

    Attributes:
        detector: The detector backend to train. Reassigned in-place by
            `fit(resume=True)` if resuming from a checkpoint, so callers
            should read `self.detector` back afterward rather than holding
            onto a separate reference to the original instance.
        output_root: Root directory under which run checkpoints are written.
    """

    def __init__(self, detector: Detector, output_root: str | Path = "runs/train") -> None:
        self.detector = detector
        self.output_root = Path(output_root)

    def fit(
        self,
        data_config: str | Path,
        epochs: int,
        run_name: str,
        image_size: int = 640,
        seed: int = 42,
        resume: bool = False,
        **train_kwargs: Any,
    ) -> dict[str, Any]:
        """Seed RNGs, train the detector, and save a `last.pt` checkpoint.

        Args:
            data_config: Path to a dataset-description YAML.
            epochs: Number of training epochs.
            run_name: Unique name for this run (used to namespace
                checkpoints under `output_root`).
            image_size: Square training image size in pixels.
            seed: Random seed for reproducibility.
            resume: If True, load weights from this run's existing
                `last.pt` checkpoint before training (continuing a
                previously interrupted run under the same `run_name`),
                instead of starting from `self.detector`'s current weights.
            **train_kwargs: Forwarded to `Detector.train`.

        Returns:
            A dict with keys `"results"` (backend training results) and
            `"checkpoints"` (the resolved `CheckpointPaths`).

        Raises:
            CheckpointError: If `resume` is True but no `last.pt` exists yet
                for `run_name`.
        """
        set_seed(seed)
        checkpoints: CheckpointPaths = resolve_checkpoint_paths(self.output_root, run_name)

        if resume:
            if not checkpoints.last.is_file():
                raise CheckpointError(
                    f"resume=True but no checkpoint found at {checkpoints.last} "
                    f"for run_name='{run_name}'. Start a fresh run (resume=False) instead."
                )
            logger.info("Resuming run '%s' from %s", run_name, checkpoints.last)
            self.detector = type(self.detector).load_checkpoint(checkpoints.last)

        logger.info(
            "Starting training run '%s': %d epochs, imgsz=%d, seed=%d%s",
            run_name,
            epochs,
            image_size,
            seed,
            " (resumed)" if resume else "",
        )
        results = self.detector.train(
            data_config=data_config,
            epochs=epochs,
            image_size=image_size,
            **train_kwargs,
        )

        self.detector.save_checkpoint(checkpoints.last)
        logger.info("Run '%s' complete. Checkpoint: %s", run_name, checkpoints.last)

        return {"results": results, "checkpoints": checkpoints}
