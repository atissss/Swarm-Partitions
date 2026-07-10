"""Unified in-memory schema that every dataset backend (VisDrone, AU-AIR,
future custom sources) parses its raw annotations into.

Keeping this schema separate from `interfaces.Detection` is deliberate:
`BoxAnnotation` is *ground truth*, produced by dataset parsers and consumed
by training/evaluation. `Detection` is *model output*, produced by
`inference` and consumed by `geotagging`. They happen to share a similar
shape, but conflating them would let a training-time concept leak into the
detector/geo-tagger contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision_model.interfaces.detection import ClassName
from vision_model.utils.exceptions import AnnotationParseError


@dataclass(frozen=True, slots=True)
class BoxAnnotation:
    """A single ground-truth bounding box annotation.

    Attributes:
        class_name: Ground-truth object category.
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        truncation: Dataset-specific truncation flag/score (0 = none).
        occlusion: Dataset-specific occlusion flag/score (0 = none).
    """

    class_name: ClassName
    bbox: tuple[float, float, float, float]
    truncation: int = 0
    occlusion: int = 0

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise AnnotationParseError(f"bbox must be a 4-tuple, got {self.bbox}")
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise AnnotationParseError(f"bbox must satisfy x1 < x2 and y1 < y2, got {self.bbox}")


@dataclass(frozen=True, slots=True)
class ImageSample:
    """One image and its associated ground-truth annotations.

    Attributes:
        image_id: Unique identifier for this sample within its dataset.
        image_path: Filesystem path to the image file.
        annotations: Ground-truth boxes present in the image.
        telemetry_timestamp: Optional ISO-8601 timestamp, populated by
            telemetry-enriched sources (e.g. AU-AIR) so downstream code can
            join against a `Telemetry` stream.
    """

    image_id: str
    image_path: Path
    annotations: list[BoxAnnotation] = field(default_factory=list)
    telemetry_timestamp: str | None = None
