"""Overlay `Detection` boxes/labels onto frames.

OpenCV is an optional-at-import-time dependency here (same lazy pattern as
`datasets.augmentations` and `models.yolo_detector`) so importing
`visualization` doesn't force an `opencv-python` install for consumers that
only need e.g. `geotagging.exporters`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_model.interfaces.detection import Detection
from vision_model.utils.exceptions import VisionModelError

# BGR colors (OpenCV convention), one per class.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (60, 180, 75),  # green
    "person": (0, 140, 255),  # orange
}
_DEFAULT_COLOR = (200, 200, 200)


def _require_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised without the optional dep
        raise VisionModelError(
            "opencv-python is required for visualization. Install it with "
            "`pip install opencv-python-headless`."
        ) from exc


def draw_detections(
    image: Any,
    detections: list[Detection],
    show_confidence: bool = True,
    thickness: int = 2,
) -> Any:
    """Draw bounding boxes and labels for `detections` onto a copy of `image`.

    Args:
        image: A BGR `np.ndarray` (OpenCV convention).
        detections: Detections to draw.
        show_confidence: If True, append the confidence score to each label.
        thickness: Box/line thickness in pixels.

    Returns:
        A new `np.ndarray` with boxes/labels drawn; `image` is not mutated.
    """
    cv2 = _require_cv2()
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = det.class_name
        if show_confidence:
            label = f"{label} {det.confidence:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - baseline - 4)),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(text_h, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(
    image: Any,
    detections: list[Detection],
    output_path: str | Path,
    show_confidence: bool = True,
) -> Path:
    """Draw `detections` onto `image` and write the result to disk.

    Args:
        image: A BGR `np.ndarray`.
        detections: Detections to draw.
        output_path: Destination image path (extension determines format).
        show_confidence: If True, append confidence scores to labels.

    Returns:
        The resolved path written to.

    Raises:
        VisionModelError: If OpenCV fails to write the output file.
    """
    cv2 = _require_cv2()
    annotated = draw_detections(image, detections, show_confidence=show_confidence)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), annotated)
    if not ok:
        raise VisionModelError(f"Failed to write annotated image to {dest}")
    return dest
