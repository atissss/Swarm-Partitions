from __future__ import annotations

from vision_model.inference.postprocess import (
    filter_by_class,
    filter_by_confidence,
    non_max_suppression,
)
from vision_model.interfaces.detection import Detection


def _det(class_name: str, confidence: float, bbox: tuple[float, float, float, float]) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(
        class_name=class_name,  # type: ignore[arg-type]
        confidence=confidence,
        bbox=bbox,
        pixel_center=((x1 + x2) / 2, (y1 + y2) / 2),
        frame_id="f",
        timestamp="2026-07-01T15:42:11Z",
    )


class TestFilterByConfidence:
    def test_keeps_at_or_above_threshold(self) -> None:
        dets = [_det("car", 0.9, (0, 0, 10, 10)), _det("car", 0.2, (0, 0, 10, 10))]
        result = filter_by_confidence(dets, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_boundary_inclusive(self) -> None:
        dets = [_det("car", 0.5, (0, 0, 10, 10))]
        assert filter_by_confidence(dets, 0.5) == dets


class TestFilterByClass:
    def test_keeps_only_allowed_classes(self) -> None:
        dets = [_det("car", 0.9, (0, 0, 10, 10)), _det("person", 0.9, (0, 0, 10, 10))]
        result = filter_by_class(dets, {"car"})
        assert len(result) == 1
        assert result[0].class_name == "car"


class TestNonMaxSuppression:
    def test_suppresses_overlapping_same_class(self) -> None:
        high = _det("car", 0.95, (0, 0, 100, 100))
        overlapping_low = _det("car", 0.60, (5, 5, 100, 100))  # heavily overlapping
        result = non_max_suppression([high, overlapping_low], iou_threshold=0.5)
        assert result == [high]

    def test_keeps_non_overlapping_boxes(self) -> None:
        a = _det("car", 0.9, (0, 0, 10, 10))
        b = _det("car", 0.8, (100, 100, 110, 110))
        result = non_max_suppression([a, b], iou_threshold=0.5)
        assert set(result) == {a, b}

    def test_does_not_suppress_across_classes(self) -> None:
        car = _det("car", 0.9, (0, 0, 100, 100))
        person = _det("person", 0.9, (0, 0, 100, 100))  # identical box, different class
        result = non_max_suppression([car, person], iou_threshold=0.5)
        assert set(result) == {car, person}

    def test_empty_input(self) -> None:
        assert non_max_suppression([], iou_threshold=0.5) == []
