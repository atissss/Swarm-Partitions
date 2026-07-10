from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vision_model.inference.predictor import Predictor
from vision_model.interfaces.detection import Detection
from vision_model.models.base import Detector
from vision_model.utils.exceptions import InferenceError


class FakeDetector(Detector):
    """Minimal in-memory `Detector` for testing `Predictor` without ultralytics."""

    def __init__(self, canned_detections: list[Detection]) -> None:
        self.canned_detections = canned_detections
        self.last_call_kwargs: dict[str, Any] | None = None

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
        self.last_call_kwargs = {
            "image": image,
            "confidence_threshold": confidence_threshold,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
        return self.canned_detections

    def save_checkpoint(self, path: str | Path) -> Path:
        return Path(path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, **kwargs: Any) -> FakeDetector:
        return cls(canned_detections=[])


def _det(class_name: str, confidence: float, bbox=(0, 0, 10, 10)) -> Detection:
    return Detection(
        class_name=class_name,  # type: ignore[arg-type]
        confidence=confidence,
        bbox=bbox,
        pixel_center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        frame_id="f",
        timestamp="2026-07-01T15:42:11Z",
    )


class TestPredictorImage:
    def test_infers_frame_id_from_path_stem(self, tmp_path: Path) -> None:
        detector = FakeDetector(canned_detections=[])
        predictor = Predictor(detector)
        image_path = tmp_path / "frame_0042.jpg"
        image_path.write_bytes(b"fake")

        predictor.predict_image(image_path, timestamp="2026-07-01T15:42:11Z")

        assert detector.last_call_kwargs["frame_id"] == "frame_0042"

    def test_array_input_requires_explicit_frame_id(self) -> None:
        detector = FakeDetector(canned_detections=[])
        predictor = Predictor(detector)
        arr = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(InferenceError):
            predictor.predict_image(arr, timestamp="2026-07-01T15:42:11Z")

    def test_postprocessing_chain_applied(self) -> None:
        low_conf = _det("car", 0.1)
        good = _det("car", 0.9, bbox=(0, 0, 20, 20))
        detector = FakeDetector(canned_detections=[low_conf, good])
        predictor = Predictor(detector, confidence_threshold=0.5, allowed_classes={"car", "person"})

        result = predictor.predict_image("img.jpg", timestamp="2026-07-01T15:42:11Z")

        assert result == [good]


class TestPredictorBatch:
    def test_mismatched_lengths_raise(self) -> None:
        detector = FakeDetector(canned_detections=[])
        predictor = Predictor(detector)
        with pytest.raises(InferenceError):
            predictor.predict_batch(["a.jpg", "b.jpg"], timestamps=["2026-07-01T15:42:11Z"])

    def test_batch_predicts_each_image(self) -> None:
        good = _det("car", 0.9, bbox=(0, 0, 20, 20))
        detector = FakeDetector(canned_detections=[good])
        predictor = Predictor(detector, confidence_threshold=0.5)

        results = predictor.predict_batch(
            ["a.jpg", "b.jpg"],
            timestamps=["2026-07-01T15:42:11Z", "2026-07-01T15:42:12Z"],
        )
        assert len(results) == 2
        assert results[0] == [good]
