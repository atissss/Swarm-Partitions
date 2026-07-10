"""Per-dataset annotation parsers.

Each parser reads a source-format annotation file and returns a list of
`BoxAnnotation` in the unified schema. This is the *only* place that needs
to know about a given dataset's raw file format — `datasets.visdrone`,
`datasets.auair`, and everything downstream only ever see `BoxAnnotation`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from vision_model.datasets.schema import BoxAnnotation
from vision_model.interfaces.detection import ClassName
from vision_model.utils.exceptions import AnnotationParseError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

# VisDrone object-category ids -> our unified class names.
# Categories not listed here (bicycle, van, truck, tricycle, awning-tricycle,
# bus, motor, others, ignored-regions) are out of scope for this pipeline
# (cars + humans only) and are dropped during parsing.
VISDRONE_CATEGORY_MAP: dict[int, ClassName] = {
    1: "person",  # pedestrian
    2: "person",  # people (non-pedestrian, e.g. seated/riding)
    4: "car",
}

# AU-AIR category names -> our unified class names. AU-AIR ships class names
# as strings rather than VisDrone's integer codes.
AUAIR_CATEGORY_MAP: dict[str, ClassName] = {
    "human": "person",
    "person": "person",
    "car": "car",
}


def parse_visdrone_annotation(annotation_path: str | Path) -> list[BoxAnnotation]:
    """Parse a single VisDrone-format annotation `.txt` file.

    VisDrone annotation lines have the format::

        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,
        <object_category>,<truncation>,<occlusion>

    Args:
        annotation_path: Path to a VisDrone `*.txt` annotation file.

    Returns:
        Parsed boxes for the "car"/"person" categories only; other VisDrone
        categories (bicycle, van, truck, bus, ignored-regions, etc.) are
        dropped since they're out of scope for this pipeline.

    Raises:
        AnnotationParseError: If the file is missing or a line is malformed.
    """
    path = Path(annotation_path)
    if not path.is_file():
        raise AnnotationParseError(f"VisDrone annotation file not found: {path}")

    annotations: list[BoxAnnotation] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            row = [c for c in row if c != ""]
            if not row:
                continue
            try:
                left, top, width, height, _score, category, truncation, occlusion = row[:8]
                category_id = int(category)
            except (ValueError, IndexError) as exc:
                raise AnnotationParseError(
                    f"{path}:{line_num}: malformed VisDrone annotation line: {row!r}"
                ) from exc

            class_name = VISDRONE_CATEGORY_MAP.get(category_id)
            if class_name is None:
                continue  # out-of-scope category (or ignored-region marker)

            x1, y1 = float(left), float(top)
            x2, y2 = x1 + float(width), y1 + float(height)
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "%s:%d: degenerate bbox skipped (w=%s, h=%s)",
                    path,
                    line_num,
                    width,
                    height,
                )
                continue

            annotations.append(
                BoxAnnotation(
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    truncation=int(truncation),
                    occlusion=int(occlusion),
                )
            )

    return annotations


def parse_auair_annotation(record: dict) -> list[BoxAnnotation]:
    """Parse a single AU-AIR JSON annotation record for one frame.

    AU-AIR ships one JSON object per frame with a `bbox` list, each entry
    holding `top`, `left`, `height`, `width`, and `class` (string category
    name), already in pixel coordinates.

    Args:
        record: One frame's annotation dict, as loaded from AU-AIR's
            `annotations.json` (a single element of the top-level list).

    Returns:
        Parsed boxes for the "car"/"person" categories only.

    Raises:
        AnnotationParseError: If `record` is missing the expected `bbox` key
            or an entry is malformed.
    """
    if "bbox" not in record:
        raise AnnotationParseError(f"AU-AIR record missing 'bbox' key: {record.keys()}")

    annotations: list[BoxAnnotation] = []
    for i, box in enumerate(record["bbox"]):
        try:
            top, left = float(box["top"]), float(box["left"])
            width, height = float(box["width"]), float(box["height"])
            raw_class = str(box["class"])
        except (KeyError, ValueError, TypeError) as exc:
            raise AnnotationParseError(f"Malformed AU-AIR bbox entry #{i}: {box!r}") from exc

        class_name = AUAIR_CATEGORY_MAP.get(raw_class.lower())
        if class_name is None:
            continue  # out-of-scope category

        x1, y1, x2, y2 = left, top, left + width, top + height
        if x2 <= x1 or y2 <= y1:
            logger.warning("Degenerate AU-AIR bbox skipped: %r", box)
            continue

        annotations.append(BoxAnnotation(class_name=class_name, bbox=(x1, y1, x2, y2)))

    return annotations
