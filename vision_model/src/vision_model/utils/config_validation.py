"""Validate a composed Hydra config before it's used to build anything.

Catches misconfiguration early and with a clear message (missing keys,
out-of-range thresholds, inconsistent class maps) instead of letting it
surface as a confusing failure deep inside `models` or `geotagging`. Every
CLI entry point calls `validate_config(cfg)` right after composing it.
"""

from __future__ import annotations

from typing import Any

from vision_model.utils.exceptions import ConfigError

_REQUIRED_TOP_LEVEL_GROUPS = ("dataset", "model", "camera", "train", "inference", "geotag")
_VALID_CLASSES = {
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
}


def _require(cfg: Any, path: str) -> Any:
    """Fetch a dotted-path key from `cfg`, raising `ConfigError` if absent."""
    node = cfg
    parts = path.split(".")
    for i, part in enumerate(parts):
        if node is None or part not in node:
            raise ConfigError(
                f"Missing required config key: '{path}' (at '{'.'.join(parts[: i + 1])}')"
            )
        node = node[part]
    return node


def _check_range(value: float, low: float, high: float, name: str) -> None:
    if not (low <= value <= high):
        raise ConfigError(f"{name} must be in [{low}, {high}], got {value}")


def _check_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def validate_config(cfg: Any) -> None:
    """Validate a fully-composed root config.

    Args:
        cfg: The Hydra-composed `DictConfig` (or any mapping-like object
            supporting `in`/`[]`/`.get`) produced from `configs/config.yaml`.

    Raises:
        ConfigError: On any missing key or out-of-range/inconsistent value.
            The message names the offending key so the fix is obvious
            without needing to read this function's source.
    """
    for group in _REQUIRED_TOP_LEVEL_GROUPS:
        if group not in cfg:
            raise ConfigError(f"Missing required config group: '{group}'")

    _validate_model(cfg)
    _validate_dataset(cfg)
    _validate_camera(cfg)
    _validate_train(cfg)
    _validate_inference(cfg)
    _validate_geotag(cfg)


def _validate_model(cfg: Any) -> None:
    class_map = _require(cfg, "model.class_map")
    if not class_map:
        raise ConfigError("model.class_map must not be empty")
    invalid = set(class_map.values()) - _VALID_CLASSES
    if invalid:
        raise ConfigError(
            f"model.class_map contains unsupported class name(s): {sorted(invalid)}. "
            f"This pipeline only supports {sorted(_VALID_CLASSES)}."
        )
    _check_range(
        float(_require(cfg, "model.confidence_threshold")), 0.0, 1.0, "model.confidence_threshold"
    )
    _check_range(float(_require(cfg, "model.iou_threshold")), 0.0, 1.0, "model.iou_threshold")


def _validate_dataset(cfg: Any) -> None:
    class_names = _require(cfg, "dataset.class_names")
    invalid = set(class_names) - _VALID_CLASSES
    if invalid:
        raise ConfigError(
            f"dataset.class_names contains unsupported class name(s): {sorted(invalid)}."
        )
    splits = _require(cfg, "dataset.splits")
    if not splits:
        raise ConfigError("dataset.splits must define at least one split")


def _validate_camera(cfg: Any) -> None:
    fx, fy = _require(cfg, "camera.intrinsics.focal_length_px")
    _check_positive(float(fx), "camera.intrinsics.focal_length_px[0]")
    _check_positive(float(fy), "camera.intrinsics.focal_length_px[1]")

    width, height = _require(cfg, "camera.intrinsics.image_size_px")
    _check_positive(float(width), "camera.intrinsics.image_size_px[0]")
    _check_positive(float(height), "camera.intrinsics.image_size_px[1]")

    cx, cy = _require(cfg, "camera.intrinsics.principal_point_px")
    if not (0 <= cx <= width):
        raise ConfigError(
            f"camera.intrinsics.principal_point_px[0]={cx} is outside image width {width}"
        )
    if not (0 <= cy <= height):
        raise ConfigError(
            f"camera.intrinsics.principal_point_px[1]={cy} is outside image height {height}"
        )


def _validate_train(cfg: Any) -> None:
    epochs = int(_require(cfg, "train.epochs"))
    if epochs <= 0:
        raise ConfigError(f"train.epochs must be positive, got {epochs}")
    image_size = int(_require(cfg, "train.image_size"))
    _check_positive(image_size, "train.image_size")
    batch = int(_require(cfg, "train.batch"))
    _check_positive(batch, "train.batch")


def _validate_inference(cfg: Any) -> None:
    _check_range(
        float(_require(cfg, "inference.confidence_threshold")),
        0.0,
        1.0,
        "inference.confidence_threshold",
    )
    _check_range(
        float(_require(cfg, "inference.iou_threshold")), 0.0, 1.0, "inference.iou_threshold"
    )
    allowed = _require(cfg, "inference.allowed_classes")
    invalid = set(allowed) - _VALID_CLASSES
    if invalid:
        raise ConfigError(
            f"inference.allowed_classes contains unsupported class(es): {sorted(invalid)}"
        )

    video_fps = cfg.inference.get("video_fps")
    if video_fps is not None:
        _check_positive(float(video_fps), "inference.video_fps")


def _validate_geotag(cfg: Any) -> None:
    max_delta = _require(cfg, "geotag.max_telemetry_delta_seconds")
    if max_delta is not None and max_delta <= 0:
        raise ConfigError(
            f"geotag.max_telemetry_delta_seconds must be positive (or null), got {max_delta}"
        )
    ground_model = _require(cfg, "geotag.ground_model")
    if ground_model not in ("flat_earth", "dtm"):
        raise ConfigError(
            f"geotag.ground_model must be 'flat_earth' or 'dtm', got '{ground_model}' "
            "('dtm' is a Phase 5 extension point and not yet implemented)."
        )
