"""Ultralytics YOLO backend for the `Detector` interface.

This is the only module in the codebase that imports `ultralytics`. Every
other package interacts with detections purely through
`interfaces.Detection`, so swapping YOLO for a different backend later means
writing a new class here (or in a sibling module) without touching
`inference`, `training`, `evaluation`, or `geotagging`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vision_model.interfaces.detection import ClassName, Detection
from vision_model.models.base import Detector
from vision_model.utils.exceptions import CheckpointError, InferenceError, ModelError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Default mapping from the trained model's class indices to this pipeline's
# class names. Must match the `names:` order in the dataset YAML the model
# was trained with (see configs/dataset/visdrone.yaml).
DEFAULT_CLASS_MAP: dict[int, ClassName] = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:  # pragma: no cover - exercised without the optional dep
        raise ModelError(
            "ultralytics is required for YOLODetector. Install it with "
            "`pip install ultralytics`."
        ) from exc


class YOLODetector(Detector):
    """`Detector` implementation wrapping an Ultralytics YOLO model.

    Attributes:
        class_map: Mapping from the underlying model's integer class index
            to this pipeline's `ClassName`. Classes not present in this map
            are silently dropped from `predict()` output (e.g. if a
            pretrained COCO checkpoint is used before fine-tuning, only its
            "car"/"person"-equivalent classes should be mapped in).
        device: Torch device string passed to Ultralytics (e.g. "cpu",
            "cuda:0", or "auto").
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        class_map: dict[int, ClassName] | None = None,
        device: str = "auto",
    ) -> None:
        YOLO = _require_ultralytics()  # noqa: N806 - matches ultralytics' class name
        self.class_map = class_map if class_map is not None else dict(DEFAULT_CLASS_MAP)
        self.device = device
        try:
            self._model = YOLO(str(model_path))
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise ModelError(f"Failed to load YOLO model from '{model_path}': {exc}") from exc
        logger.info("Loaded YOLO model from %s (device=%s)", model_path, device)

    def train(
        self,
        data_config: str | Path,
        epochs: int,
        image_size: int = 640,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train/fine-tune the underlying YOLO model.

        Args:
            data_config: Path to an Ultralytics-format data YAML (see
                `configs/dataset/visdrone.yaml` for the expected shape).
            epochs: Number of training epochs.
            image_size: Square training image size in pixels.
            **kwargs: Forwarded to `ultralytics.YOLO.train` (e.g. `batch`,
                `lr0`, `patience`, `seed`).

        Returns:
            A dict summary of the Ultralytics training results.

        Raises:
            ModelError: If training fails.
        """
        try:
            results = self._model.train(
                data=str(data_config),
                epochs=epochs,
                imgsz=image_size,
                device=self.device,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"YOLO training failed: {exc}") from exc
        logger.info("Training complete: %s epochs on %s", epochs, data_config)
        return {"results": results}

    def validate(self, data_config: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Run Ultralytics' built-in validation.

        Args:
            data_config: Path to an Ultralytics-format data YAML.
            **kwargs: Forwarded to `ultralytics.YOLO.val`.

        Returns:
            A dict summary of Ultralytics' validation metrics. For the
            pipeline's own standardized mAP/precision/recall/F1 numbers, use
            `evaluation.metrics` against raw `Detection` output instead.

        Raises:
            ModelError: If validation fails.
        """
        try:
            metrics = self._model.val(data=str(data_config), device=self.device, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"YOLO validation failed: {exc}") from exc
        return {"metrics": metrics}

    def predict(
        self,
        image: str | Path | np.ndarray,
        confidence_threshold: float = 0.25,
        frame_id: str = "",
        timestamp: str = "",
    ) -> list[Detection]:
        """Run YOLO inference on a single image.

        Args:
            image: Path to an image file, or an already-loaded `np.ndarray`
                in BGR order (OpenCV convention), as expected by Ultralytics.
            confidence_threshold: Minimum confidence to keep a detection.
            frame_id: Identifier stamped onto every returned `Detection`.
            timestamp: ISO-8601 timestamp stamped onto every returned
                `Detection`.

        Returns:
            Detections for classes present in `self.class_map`, already
            filtered by `confidence_threshold`.

        Raises:
            InferenceError: If the underlying model call fails.
        """
        try:
            results = self._model.predict(
                source=image,
                conf=confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"YOLO inference failed: {exc}") from exc

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                class_name = self.class_map.get(cls_id)
                if class_name is None:
                    continue  # class outside this pipeline's scope

                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        pixel_center=(cx, cy),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        return detections

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save the current model weights via Ultralytics' exporter.

        Args:
            path: Destination `.pt` file path.

        Returns:
            The resolved path written to.

        Raises:
            CheckpointError: If saving fails.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._model.save(str(dest))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointError(f"Failed to save checkpoint to '{dest}': {exc}") from exc
        logger.info("Saved checkpoint to %s", dest)
        return dest

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        class_map: dict[int, ClassName] | None = None,
        device: str = "auto",
    ) -> YOLODetector:
        """Load a `YOLODetector` from a saved `.pt` checkpoint.

        Args:
            path: Path to a checkpoint previously written by
                `save_checkpoint`, or any Ultralytics-compatible weights file.
            class_map: Class-index -> `ClassName` mapping to use.
            device: Torch device string.

        Returns:
            A ready-to-use `YOLODetector`.

        Raises:
            CheckpointError: If the checkpoint file does not exist.
        """
        if not Path(path).is_file():
            raise CheckpointError(f"Checkpoint not found: {path}")
        return cls(model_path=path, class_map=class_map, device=device)
