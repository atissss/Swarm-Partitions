"""Name -> detector class registry, mirroring `datasets.registry`."""

from __future__ import annotations

from typing import Any

from vision_model.models.base import Detector
from vision_model.utils.exceptions import ConfigError

_REGISTRY: dict[str, type[Detector]] = {}


def register(name: str, detector_cls: type[Detector]) -> None:
    """Register a `Detector` class under `name`.

    Args:
        name: Lookup key used in model configs (case-insensitive).
        detector_cls: `Detector` subclass to register.

    Raises:
        ConfigError: If `name` is already registered to a different class.
    """
    key = name.lower()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not detector_cls:
        raise ConfigError(
            f"Detector name '{name}' is already registered to {existing.__name__}; "
            f"cannot re-register to {detector_cls.__name__}."
        )
    _REGISTRY[key] = detector_cls


def build_detector(name: str, **kwargs: Any) -> Detector:
    """Construct a registered detector by name.

    Args:
        name: Registered detector name, e.g. `"yolo"`.
        **kwargs: Forwarded to the detector class constructor.

    Returns:
        An instantiated `Detector` subclass.

    Raises:
        ConfigError: If `name` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ConfigError(f"Unknown detector '{name}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def available_detectors() -> list[str]:
    """List all registered detector names."""
    return sorted(_REGISTRY)


def _register_builtins() -> None:
    """Lazily register built-in detectors.

    Deferred (rather than a top-level import) so importing `models.registry`
    doesn't force an `ultralytics` import — and therefore doesn't fail — in
    environments that only need e.g. `available_detectors()` or where
    ultralytics isn't installed yet.
    """
    from vision_model.models.yolo_detector import YOLODetector

    register("yolo", YOLODetector)


try:
    _register_builtins()
except Exception:  # noqa: BLE001 - ultralytics not installed; registry stays usable
    pass
