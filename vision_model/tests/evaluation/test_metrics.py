from __future__ import annotations

import pytest

from vision_model.datasets.schema import BoxAnnotation
from vision_model.evaluation.metrics import evaluate
from vision_model.interfaces.detection import Detection


def _det(class_name: str, confidence: float, bbox) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(
        class_name=class_name,  # type: ignore[arg-type]
        confidence=confidence,
        bbox=bbox,
        pixel_center=((x1 + x2) / 2, (y1 + y2) / 2),
        frame_id="f",
        timestamp="2026-07-01T15:42:11Z",
    )


def _gt(class_name: str, bbox) -> BoxAnnotation:
    return BoxAnnotation(class_name=class_name, bbox=bbox)  # type: ignore[arg-type]


class TestEvaluate:
    def test_perfect_predictions_yield_perfect_metrics(self) -> None:
        gt_box = (0.0, 0.0, 100.0, 100.0)
        predictions = {"img1": [_det("car", 0.99, gt_box)]}
        ground_truth = {"img1": [_gt("car", gt_box)]}

        report = evaluate(predictions, ground_truth, class_names=["car"], iou_threshold=0.5)

        car_metrics = report.per_class["car"]
        assert car_metrics.precision == 1.0
        assert car_metrics.recall == 1.0
        assert car_metrics.f1 == 1.0
        assert car_metrics.average_precision == pytest.approx(1.0)
        assert report.mean_average_precision == pytest.approx(1.0)

    def test_missed_detection_reduces_recall(self) -> None:
        predictions = {"img1": []}
        ground_truth = {"img1": [_gt("car", (0, 0, 100, 100))]}

        report = evaluate(predictions, ground_truth, class_names=["car"], iou_threshold=0.5)

        car_metrics = report.per_class["car"]
        assert car_metrics.recall == 0.0
        assert car_metrics.num_ground_truth == 1
        assert car_metrics.num_predictions == 0

    def test_false_positive_reduces_precision(self) -> None:
        predictions = {"img1": [_det("car", 0.9, (200, 200, 300, 300))]}  # nowhere near GT
        ground_truth = {"img1": [_gt("car", (0, 0, 100, 100))]}

        report = evaluate(predictions, ground_truth, class_names=["car"], iou_threshold=0.5)

        car_metrics = report.per_class["car"]
        assert car_metrics.precision == 0.0
        assert car_metrics.recall == 0.0

    def test_no_ground_truth_or_predictions_is_zero_not_nan(self) -> None:
        report = evaluate({}, {}, class_names=["car"], iou_threshold=0.5)
        car_metrics = report.per_class["car"]
        assert car_metrics.precision == 0.0
        assert car_metrics.recall == 0.0
        assert car_metrics.f1 == 0.0
        assert car_metrics.average_precision == 0.0

    def test_per_class_isolation(self) -> None:
        # A perfect "car" prediction shouldn't affect "person" metrics.
        predictions = {
            "img1": [_det("car", 0.9, (0, 0, 100, 100)), _det("person", 0.9, (500, 500, 550, 600))]
        }
        ground_truth = {"img1": [_gt("car", (0, 0, 100, 100)), _gt("person", (0, 0, 10, 10))]}

        report = evaluate(predictions, ground_truth, class_names=["car", "person"])

        assert report.per_class["car"].recall == 1.0
        assert report.per_class["person"].recall == 0.0
