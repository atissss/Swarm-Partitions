"""Frame -> `Detection` orchestration.

`Predictor` is the seam between a concrete `Detector` backend and the rest
of the pipeline: it calls the detector, then applies the standardized
post-processing chain (confidence gating, class filtering, NMS) so behavior
is consistent regardless of which backend produced the raw detections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_model.inference.postprocess import (
    filter_by_class,
    filter_by_confidence,
    non_max_suppression,
)
from vision_model.interfaces.detection import Detection
from vision_model.models.base import Detector
from vision_model.utils.exceptions import InferenceError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)


class Predictor:
    """Runs a `Detector` on images and applies standardized post-processing.

    Attributes:
        detector: The underlying detection backend.
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for the NMS pass.
        allowed_classes: If set, only these class names are kept (defaults
            to `{"car", "person"}`, the pipeline's full scope).
    """

    def __init__(
        self,
        detector: Detector,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        allowed_classes: set[str] | None = None,
    ) -> None:
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.allowed_classes = allowed_classes or {"car", "person"}

    def predict_image(
        self,
        image: str | Path | np.ndarray,
        timestamp: str,
        frame_id: str | None = None,
    ) -> list[Detection]:
        """Run detection + post-processing on a single image.

        Args:
            image: Path to an image file, or a pre-loaded `np.ndarray`.
            timestamp: ISO-8601 UTC timestamp of the source frame. Required
                (rather than defaulted to "now") because an inaccurate
                timestamp silently produces an inaccurate geo-tag downstream
                — better to fail loudly if the caller has no real timestamp
                to provide.
            frame_id: Identifier for the source frame. Defaults to the
                image's file stem if `image` is a path, otherwise must be
                provided explicitly.

        Returns:
            Post-processed detections (confidence-gated, class-filtered,
            NMS-applied), each with `class_name`, `bbox`, `pixel_center`,
            `frame_id`, and `timestamp` populated. Geographic fields are
            not set here — that's `geotagging`'s job.

        Raises:
            InferenceError: If `frame_id` cannot be inferred and wasn't
                provided, or if the underlying detector call fails.
        """
        if frame_id is None:
            if isinstance(image, (str, Path)):
                frame_id = Path(image).stem
            else:
                raise InferenceError(
                    "frame_id must be provided explicitly when `image` is an in-memory array."
                )

        raw_detections = self.detector.predict(
            image,
            confidence_threshold=self.confidence_threshold,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        detections = filter_by_confidence(raw_detections, self.confidence_threshold)
        detections = filter_by_class(detections, self.allowed_classes)
        detections = non_max_suppression(detections, self.iou_threshold)

        logger.debug(
            "predict_image(%s): %d raw -> %d after post-processing",
            frame_id,
            len(raw_detections),
            len(detections),
        )
        return detections

    def predict_batch(
        self,
        images: list[str | Path],
        timestamps: list[str],
        frame_ids: list[str] | None = None,
    ) -> list[list[Detection]]:
        """Run `predict_image` over a batch of images.

        Args:
            images: Paths to image files.
            timestamps: ISO-8601 timestamps, one per image, same order.
            frame_ids: Optional frame identifiers, one per image. Defaults
                to each image's file stem.

        Returns:
            One `list[Detection]` per input image, same order.

        Raises:
            InferenceError: If `images` and `timestamps` (or `frame_ids`,
                when provided) have mismatched lengths.
        """
        if len(images) != len(timestamps):
            raise InferenceError(
                f"images ({len(images)}) and timestamps ({len(timestamps)}) length mismatch."
            )
        if frame_ids is not None and len(frame_ids) != len(images):
            raise InferenceError(
                f"images ({len(images)}) and frame_ids ({len(frame_ids)}) length mismatch."
            )

        results: list[list[Detection]] = []
        for i, image in enumerate(images):
            fid = frame_ids[i] if frame_ids is not None else None
            results.append(self.predict_image(image, timestamp=timestamps[i], frame_id=fid))
        return results
