"""Detection metrics: Precision, Recall, F1, and mAP.

Operates purely on `Detection` (predictions) vs `BoxAnnotation` (ground
truth), so it's usable both against live model output and against
previously-saved detection exports, without needing a `Detector` instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from vision_model.datasets.schema import BoxAnnotation
from vision_model.inference.postprocess import compute_iou
from vision_model.interfaces.detection import Detection


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision/recall/F1/AP for a single class at a fixed IoU threshold.

    Attributes:
        class_name: The class these metrics were computed for.
        precision: TP / (TP + FP) using each prediction's best-confidence
            greedy match to an unused ground-truth box.
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall (0 if both are 0).
        average_precision: 11-point-interpolated average precision at the
            configured IoU threshold.
        num_ground_truth: Total ground-truth boxes for this class.
        num_predictions: Total predicted boxes for this class.
    """

    class_name: str
    precision: float
    recall: float
    f1: float
    average_precision: float
    num_ground_truth: int
    num_predictions: int


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate evaluation results across all classes.

    Attributes:
        per_class: `ClassMetrics` for each class present in ground truth
            and/or predictions.
        mean_average_precision: Mean of `average_precision` across classes
            (this pipeline's "mAP" at the configured IoU threshold — a
            single-IoU mAP, i.e. mAP@`iou_threshold`, not the COCO
            mAP@[.5:.95] sweep).
        iou_threshold: The IoU threshold matches were computed at.
    """

    per_class: dict[str, ClassMetrics]
    mean_average_precision: float
    iou_threshold: float


def _match_predictions_to_ground_truth(
    predictions: list[Detection],
    ground_truth: list[BoxAnnotation],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedily match predictions (by descending confidence) to unused GT boxes.

    Returns:
        (true_positives, false_positives, false_negatives) counts.
    """
    ordered_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    matched_gt = [False] * len(ground_truth)

    tp = 0
    for pred in ordered_preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(ground_truth):
            if matched_gt[i]:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            tp += 1

    fp = len(predictions) - tp
    fn = len(ground_truth) - sum(matched_gt)
    return tp, fp, fn


def _average_precision(
    predictions: list[Detection],
    ground_truth: list[BoxAnnotation],
    iou_threshold: float,
) -> float:
    """11-point interpolated average precision for one class."""
    if not ground_truth:
        return 0.0
    if not predictions:
        return 0.0

    ordered_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    matched_gt = [False] * len(ground_truth)

    precisions: list[float] = []
    recalls: list[float] = []
    tp_count = 0
    for i, pred in enumerate(ordered_preds, start=1):
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(ground_truth):
            if matched_gt[j]:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, j

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            tp_count += 1

        precisions.append(tp_count / i)
        recalls.append(tp_count / len(ground_truth))

    ap = 0.0
    for recall_level in (r / 10 for r in range(11)):
        precisions_at_recall = [
            p for p, r in zip(precisions, recalls, strict=True) if r >= recall_level
        ]
        ap += (max(precisions_at_recall) if precisions_at_recall else 0.0) / 11.0
    return ap


def evaluate(
    predictions_by_image: dict[str, list[Detection]],
    ground_truth_by_image: dict[str, list[BoxAnnotation]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Compute precision/recall/F1/mAP across a validation set.

    Args:
        predictions_by_image: Predicted detections, keyed by `image_id`
            (same keys as `ground_truth_by_image`).
        ground_truth_by_image: Ground-truth boxes, keyed by `image_id`.
        class_names: Classes to report metrics for, e.g. `["car", "person"]`.
        iou_threshold: IoU threshold for counting a prediction as a true
            positive.

    Returns:
        An `EvaluationReport` with per-class and mean metrics.
    """
    per_class: dict[str, ClassMetrics] = {}

    for class_name in class_names:
        tp_total = fp_total = fn_total = 0
        gt_count = pred_count = 0
        ap_numerator: list[float] = []

        for image_id in ground_truth_by_image:
            preds = [
                d for d in predictions_by_image.get(image_id, []) if d.class_name == class_name
            ]
            gts = [a for a in ground_truth_by_image.get(image_id, []) if a.class_name == class_name]
            gt_count += len(gts)
            pred_count += len(preds)

            tp, fp, fn = _match_predictions_to_ground_truth(preds, gts, iou_threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            ap_numerator.append(_average_precision(preds, gts, iou_threshold) * max(len(gts), 0))

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap = sum(ap_numerator) / gt_count if gt_count > 0 else 0.0

        per_class[class_name] = ClassMetrics(
            class_name=class_name,
            precision=precision,
            recall=recall,
            f1=f1,
            average_precision=ap,
            num_ground_truth=gt_count,
            num_predictions=pred_count,
        )

    map_score = (
        sum(m.average_precision for m in per_class.values()) / len(per_class) if per_class else 0.0
    )
    return EvaluationReport(
        per_class=per_class, mean_average_precision=map_score, iou_threshold=iou_threshold
    )
