"""Shared test doubles.

`FakeDetector` lets tests exercise `Predictor`/`Trainer`/batch-runner logic
without needing `torch`/`ultralytics` installed or a real trained model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vision_model.interfaces.detection import Detection
from vision_model.models.base import Detector


class FakeDetector(Detector):
    """In-memory `Detector` returning a fixed, caller-supplied detection list.

    Every `predict()` call returns `canned_detections`, stamped with the
    `frame_id`/`timestamp` passed to that call (so callers can distinguish
    detections across frames even though the "model" is static).
    """

    def __init__(self, canned_detections: list[Detection] | None = None) -> None:
        self.canned_detections = canned_detections if canned_detections is not None else []
        self.predict_calls: list[dict[str, Any]] = []

    def train(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def validate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def predict(
        self,
        image: str | Path | np.ndarray,
        confidence_threshold: float = 0.25,
        frame_id: str = "",
        timestamp: str = "",
    ) -> list[Detection]:
        self.predict_calls.append(
            {
                "image": image,
                "confidence_threshold": confidence_threshold,
                "frame_id": frame_id,
                "timestamp": timestamp,
            }
        )
        return [
            Detection(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                pixel_center=d.pixel_center,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            for d in self.canned_detections
        ]

    def save_checkpoint(self, path: str | Path) -> Path:
        return Path(path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, **kwargs: Any) -> FakeDetector:
        return cls()
