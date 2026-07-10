"""Albumentations-based augmentation pipelines for aerial imagery.

Aerial/drone imagery favors augmentations that reflect real flight
variation (altitude-driven scale changes, rotation since drones have no
fixed "up", brightness swings from sun angle) over augmentations tuned for
ground-level photography (e.g. aggressive perspective warps).

Albumentations is an optional heavy dependency: importing this module
without it installed raises a clear `VisionModelError` only when a builder
function is actually called, not at import time, so the rest of the
`datasets` package stays usable without it (e.g. for annotation parsing or
inference-only workflows).
"""

from __future__ import annotations

from typing import Any

from vision_model.utils.exceptions import VisionModelError


def _require_albumentations() -> Any:
    try:
        import albumentations as A  # noqa: N812 - "A" is albumentations' documented alias

        return A
    except ImportError as exc:  # pragma: no cover - exercised without the optional dep
        raise VisionModelError(
            "albumentations is required for augmentation pipelines. "
            "Install it with `pip install albumentations`."
        ) from exc


def build_train_transform(image_size: int = 640) -> Any:
    """Build the training-time augmentation pipeline.

    Args:
        image_size: Square image size (pixels) to resize to before other
            augmentations, matching the detector's expected input size.

    Returns:
        An `albumentations.Compose` pipeline configured with
        `bbox_params` for Pascal-VOC-style `(x1, y1, x2, y2)` boxes plus a
        `class_labels` field, matching `BoxAnnotation`.
    """
    A = _require_albumentations()  # noqa: N806 - conventional albumentations alias
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.GaussNoise(p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.2
        ),
    )


def build_eval_transform(image_size: int = 640) -> Any:
    """Build the validation/inference-time transform (resize + pad only).

    Args:
        image_size: Square image size (pixels) to resize to.

    Returns:
        An `albumentations.Compose` pipeline with no randomness, so
        evaluation/inference is deterministic.
    """
    A = _require_albumentations()  # noqa: N806 - conventional albumentations alias
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
