"""Post-processing over raw `Detection` lists.

Ultralytics YOLO already applies its own NMS internally, but this module
exists so the pipeline isn't locked into trusting a specific backend's
post-processing — e.g. when merging detections from multiple models/passes,
or re-thresholding already-computed detections without re-running inference.
All functions are pure (`list[Detection] -> list[Detection]`) and therefore
trivially unit-testable without a model.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vision_model.interfaces.detection import Detection


def filter_by_confidence(detections: list[Detection], threshold: float) -> list[Detection]:
    """Keep only detections with `confidence >= threshold`.

    Args:
        detections: Input detections.
        threshold: Minimum confidence to keep, in [0, 1].

    Returns:
        Filtered detections, order preserved.
    """
    return [d for d in detections if d.confidence >= threshold]


def filter_by_class(detections: list[Detection], allowed_classes: set[str]) -> list[Detection]:
    """Keep only detections whose class is in `allowed_classes`.

    Args:
        detections: Input detections.
        allowed_classes: Set of class names to keep, e.g. `{"car"}`.

    Returns:
        Filtered detections, order preserved.
    """
    return [d for d in detections if d.class_name in allowed_classes]


def compute_iou(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes.

    Args:
        box_a: First box as (x1, y1, x2, y2).
        box_b: Second box as (x1, y1, x2, y2).

    Returns:
        IoU in [0, 1]. 0 if the boxes don't overlap.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Apply per-class greedy NMS over a list of detections.

    Args:
        detections: Input detections, any order.
        iou_threshold: Boxes of the same class with IoU above this
            threshold are suppressed in favor of the higher-confidence box.

    Returns:
        Surviving detections, sorted by descending confidence within each
        class as a side effect of the greedy algorithm, then re-merged
        across classes in original relative order.
    """
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_class[d.class_name].append(d)

    kept: list[Detection] = []
    for class_dets in by_class.values():
        ordered = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        suppressed = np.zeros(len(ordered), dtype=bool)
        for i, det_i in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                if compute_iou(det_i.bbox, ordered[j].bbox) > iou_threshold:
                    suppressed[j] = True

    return kept
