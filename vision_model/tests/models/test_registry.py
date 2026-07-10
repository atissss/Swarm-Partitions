from __future__ import annotations

import pytest

from vision_model.models.registry import available_detectors, build_detector, register
from vision_model.utils.exceptions import ConfigError


class TestModelRegistry:
    def test_yolo_registered_by_default(self) -> None:
        assert "yolo" in available_detectors()

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ConfigError):
            build_detector("not_a_real_detector")

    def test_re_registering_same_class_is_a_noop(self) -> None:
        from vision_model.models.yolo_detector import YOLODetector

        register("yolo", YOLODetector)  # should not raise
        assert "yolo" in available_detectors()

    def test_re_registering_different_class_raises(self) -> None:
        from vision_model.models.base import Detector

        class _Other(Detector):
            def train(self, *a, **k):
                return {}

            def validate(self, *a, **k):
                return {}

            def predict(self, *a, **k):
                return []

            def save_checkpoint(self, path):
                return path

            @classmethod
            def load_checkpoint(cls, path, **k):
                return cls()

        with pytest.raises(ConfigError):
            register("yolo", _Other)
