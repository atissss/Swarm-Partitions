"""Evaluate a `Predictor` against a dataset's ground truth.

Bridges `datasets` (ground truth) and `inference` (predictions) into the
pure `evaluation.metrics.evaluate` function, without either of those
packages needing to know about the other.
"""

from __future__ import annotations

from vision_model.datasets.base import AerialDetectionDataset
from vision_model.evaluation.metrics import EvaluationReport, evaluate
from vision_model.inference.predictor import Predictor
from vision_model.interfaces.detection import Detection
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

# A placeholder timestamp for images that have no real capture time (e.g.
# VisDrone, which ships no telemetry). Evaluation only needs pixel-space
# boxes, so the exact value here is inert.
_NO_TIMESTAMP = "1970-01-01T00:00:00Z"


def evaluate_dataset(
    predictor: Predictor,
    dataset: AerialDetectionDataset,
    class_names: list[str] | None = None,
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Run `predictor` over every image in `dataset` and compute metrics.

    Args:
        predictor: Configured `Predictor` to run inference with.
        dataset: Dataset providing images and ground-truth annotations.
        class_names: Classes to report metrics for. Defaults to
            `["car", "person"]`.
        iou_threshold: IoU threshold for counting a true positive.

    Returns:
        An `EvaluationReport` aggregated across the whole dataset.
    """
    class_names = class_names or ["car", "person"]

    predictions_by_image: dict[str, list[Detection]] = {}
    ground_truth_by_image: dict[str, list] = {}

    for sample in dataset.index:
        timestamp = sample.telemetry_timestamp or _NO_TIMESTAMP
        detections = predictor.predict_image(
            sample.image_path, timestamp=timestamp, frame_id=sample.image_id
        )
        predictions_by_image[sample.image_id] = detections
        ground_truth_by_image[sample.image_id] = sample.annotations

    logger.info("Evaluating %d images at IoU=%.2f", len(dataset), iou_threshold)
    return evaluate(predictions_by_image, ground_truth_by_image, class_names, iou_threshold)
