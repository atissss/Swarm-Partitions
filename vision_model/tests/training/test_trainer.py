from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vision_model.interfaces.detection import Detection
from vision_model.models.base import Detector
from vision_model.training.trainer import Trainer
from vision_model.utils.exceptions import CheckpointError


class RecordingDetector(Detector):
    """Fake detector that records train() calls and checkpoint saves/loads."""

    def __init__(self, loaded_from: Path | None = None) -> None:
        self.train_calls: list[dict[str, Any]] = []
        self.saved_to: Path | None = None
        self.loaded_from = loaded_from

    def train(self, data_config, epochs, image_size: int = 640, **kwargs: Any) -> dict[str, Any]:
        self.train_calls.append(
            {"data_config": data_config, "epochs": epochs, "image_size": image_size, **kwargs}
        )
        return {"map50": 0.42}

    def validate(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {}

    def predict(self, *a: Any, **k: Any) -> list[Detection]:
        return []

    def save_checkpoint(self, path) -> Path:
        self.saved_to = Path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        return Path(path)

    @classmethod
    def load_checkpoint(cls, path, **kwargs: Any) -> RecordingDetector:
        return cls(loaded_from=Path(path))


class TestTrainer:
    def test_fit_delegates_to_detector_and_saves_checkpoint(self, tmp_path: Path) -> None:
        detector = RecordingDetector()
        trainer = Trainer(detector, output_root=tmp_path / "runs")

        result = trainer.fit(
            data_config="configs/dataset/visdrone.yaml",
            epochs=5,
            run_name="exp001",
            image_size=512,
            seed=42,
        )

        assert len(detector.train_calls) == 1
        assert detector.train_calls[0]["epochs"] == 5
        assert detector.train_calls[0]["image_size"] == 512
        assert detector.saved_to == result["checkpoints"].last
        assert detector.saved_to.is_file()
        assert result["results"] == {"map50": 0.42}

    def test_fit_is_reproducible_given_same_seed(self, tmp_path: Path) -> None:
        import random

        detector = RecordingDetector()
        trainer = Trainer(detector, output_root=tmp_path / "runs")

        trainer.fit(data_config="d.yaml", epochs=1, run_name="a", seed=99)
        first_draw = random.random()

        trainer.fit(data_config="d.yaml", epochs=1, run_name="b", seed=99)
        second_draw = random.random()

        assert first_draw == second_draw

    def test_resume_without_existing_checkpoint_raises(self, tmp_path: Path) -> None:
        detector = RecordingDetector()
        trainer = Trainer(detector, output_root=tmp_path / "runs")

        with pytest.raises(CheckpointError):
            trainer.fit(data_config="d.yaml", epochs=1, run_name="never_run", resume=True)

    def test_resume_loads_checkpoint_before_training(self, tmp_path: Path) -> None:
        detector = RecordingDetector()
        trainer = Trainer(detector, output_root=tmp_path / "runs")

        # First run creates last.pt
        first_result = trainer.fit(data_config="d.yaml", epochs=1, run_name="exp001")
        checkpoint_path = first_result["checkpoints"].last
        assert checkpoint_path.is_file()

        # Second call with resume=True should load from that checkpoint
        trainer.fit(data_config="d.yaml", epochs=1, run_name="exp001", resume=True)

        assert trainer.detector.loaded_from == checkpoint_path
