"""Stable detector output contract.

`Detection` is the only object the `inference` package is allowed to produce
and the only object the `geotagging` package is allowed to consume as its
detector-side input. It intentionally carries no geographic information —
that is populated later, by `geotagging`, into a separate `GeoTaggedDetection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClassName = Literal["car", "person"]


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object in image/pixel space.

    Attributes:
        class_name: Detected object category. Restricted to the classes this
            pipeline is trained for ("car", "person").
        confidence: Model confidence score in [0.0, 1.0].
        bbox: Axis-aligned bounding box as (x1, y1, x2, y2) in pixel
            coordinates, with (x1, y1) the top-left and (x2, y2) the
            bottom-right corner.
        pixel_center: (cx, cy) center of the bounding box in pixel
            coordinates. Stored explicitly (rather than derived) so that
            geo-tagging can use a detector-specified anchor point (e.g. the
            bottom-center of a bounding box for a standing person) if a
            future detector chooses a different convention.
        frame_id: Identifier of the source frame or video the detection came
            from (e.g. file name, or "<video_id>:<frame_index>").
        timestamp: ISO-8601 UTC timestamp of the source frame, used to match
            this detection against the nearest telemetry sample.

    Raises:
        ValueError: If `confidence` is outside [0, 1], if `bbox` is not a
            4-tuple with x1 < x2 and y1 < y2, or if `pixel_center` is not a
            2-tuple.
    """

    class_name: ClassName
    confidence: float
    bbox: tuple[float, float, float, float]
    pixel_center: tuple[float, float]
    frame_id: str
    timestamp: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must be a 4-tuple (x1, y1, x2, y2), got {self.bbox}")
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"bbox must satisfy x1 < x2 and y1 < y2, got {self.bbox}")
        if len(self.pixel_center) != 2:
            raise ValueError(f"pixel_center must be a 2-tuple (cx, cy), got {self.pixel_center}")

    @property
    def width(self) -> float:
        """Bounding box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Bounding box area in square pixels."""
        return self.width * self.height
