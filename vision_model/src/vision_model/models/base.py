"""Base detector interface every detection backend implements.

`Detector` is intentionally framework-agnostic at the type level (it only
promises a `predict(image) -> list[Detection]` contract plus lifecycle
methods) so `inference`, `training`, and `evaluation` never need to know
whether the concrete backend is Ultralytics YOLO, a future RT-DETR wrapper,
or anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from vision_model.interfaces.detection import Detection


class Detector(ABC):
    """Abstract base for all object-detection model backends."""

    @abstractmethod
    def train(
        self,
        data_config: str | Path,
        epochs: int,
        image_size: int = 640,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train (or fine-tune) the detector.

        Args:
            data_config: Path to a dataset-description YAML in the backend's
                expected format (e.g. Ultralytics data YAML).
            epochs: Number of training epochs.
            image_size: Square training image size in pixels.
            **kwargs: Backend-specific training options.

        Returns:
            A dict of training results/metrics as reported by the backend.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, data_config: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Run validation and return backend-reported metrics.

        Args:
            data_config: Path to a dataset-description YAML.
            **kwargs: Backend-specific validation options.

        Returns:
            A dict of validation metrics (backend-specific keys; use
            `evaluation.metrics` for the pipeline's own standardized
            mAP/precision/recall/F1 computation instead of relying on
            backend-specific formats where consistency matters).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: str | Path | np.ndarray,
        confidence_threshold: float = 0.25,
        frame_id: str = "",
        timestamp: str = "",
    ) -> list[Detection]:
        """Run inference on a single image and return unified `Detection` objects.

        Args:
            image: Path to an image file, or an already-loaded BGR/RGB
                `np.ndarray` (backend-specific convention documented on the
                concrete class).
            confidence_threshold: Minimum confidence to keep a detection.
            frame_id: Identifier to stamp onto every returned `Detection`.
            timestamp: ISO-8601 timestamp to stamp onto every returned
                `Detection`.

        Returns:
            Detections for the "car"/"person" classes, already filtered by
            `confidence_threshold` and any backend-side NMS.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist the current model weights.

        Args:
            path: Destination path for the checkpoint.

        Returns:
            The resolved path written to.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str | Path, **kwargs: Any) -> Detector:
        """Construct a `Detector` from a saved checkpoint.

        Args:
            path: Path to a checkpoint previously written by
                `save_checkpoint`.
            **kwargs: Backend-specific loading options.

        Returns:
            A ready-to-use `Detector` instance.
        """
        raise NotImplementedError
