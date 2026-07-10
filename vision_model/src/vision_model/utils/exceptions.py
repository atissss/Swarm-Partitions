"""Shared exception hierarchy for the vision_model package.

Every package raises typed exceptions rooted here instead of bare
`Exception`/`ValueError`, so callers can catch precisely at whatever level
they need (e.g. catch all `GeoTagError`s but let `DatasetError`s propagate).
"""

from __future__ import annotations


class VisionModelError(Exception):
    """Base class for all vision_model errors."""


class DatasetError(VisionModelError):
    """Raised for dataset loading or annotation parsing failures."""


class AnnotationParseError(DatasetError):
    """Raised when an annotation file cannot be parsed into the unified schema."""


class ModelError(VisionModelError):
    """Raised for model construction, training, or checkpoint failures."""


class CheckpointError(ModelError):
    """Raised when a checkpoint cannot be saved or loaded."""


class InferenceError(VisionModelError):
    """Raised for detector inference failures."""


class GeoTagError(VisionModelError):
    """Raised when a detection cannot be geo-tagged."""


class TelemetryNotFoundError(GeoTagError):
    """Raised when no telemetry sample is available near a detection's timestamp."""


class ProjectionError(GeoTagError):
    """Raised when a camera ray cannot be intersected with the ground.

    E.g. the ray is parallel to (or points away from) the ground plane —
    typically a detection near the horizon or invalid camera attitude.
    """


class ConfigError(VisionModelError):
    """Raised for configuration loading or validation failures."""
