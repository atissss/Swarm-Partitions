"""AU-AIR dataset backend.

AU-AIR ships a single `annotations.json` file per split: a JSON list of
per-frame records, each holding an image file name, a `bbox` list (parsed by
`annotation_parsers.parse_auair_annotation`), and flight telemetry fields
(GPS, altitude, timestamp) captured at the moment the frame was taken. That
per-frame timestamp is threaded through into `ImageSample.telemetry_timestamp`
so it can later be joined against a `Telemetry` stream by `geotagging`.

Expected layout::

    <root>/
        images/
            frame_000001.jpg
            ...
        annotations.json
"""

from __future__ import annotations

from pathlib import Path

from vision_model.datasets.annotation_parsers import parse_auair_annotation
from vision_model.datasets.base import AerialDetectionDataset
from vision_model.datasets.schema import ImageSample
from vision_model.utils.exceptions import DatasetError
from vision_model.utils.io import read_json
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AUAIRDataset(AerialDetectionDataset):
    """Aerial detection dataset backend for AU-AIR.

    Attributes:
        images_dir: Resolved `<root>/images` directory.
        annotations_path: Resolved `<root>/annotations.json` file.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        self.images_dir = Path(root) / "images"
        self.annotations_path = Path(root) / "annotations.json"
        super().__init__(root=root, split=split, transform=transform)

    def load_index(self) -> list[ImageSample]:
        """Parse `annotations.json` and build the sample index.

        Returns:
            One `ImageSample` per record in `annotations.json` whose image
            file exists on disk under `images_dir`.

        Raises:
            DatasetError: If `annotations_path` does not exist or is not a
                JSON list of records.
        """
        if not self.annotations_path.is_file():
            raise DatasetError(f"AU-AIR annotations file not found: {self.annotations_path}")

        records = read_json(self.annotations_path)
        if not isinstance(records, list):
            raise DatasetError(
                f"Expected {self.annotations_path} to contain a JSON list, "
                f"got {type(records).__name__}"
            )

        samples: list[ImageSample] = []
        for record in records:
            image_name = record.get("image_name")
            if not image_name:
                logger.warning("Skipping AU-AIR record with no 'image_name': %s", record)
                continue

            image_path = self.images_dir / image_name
            if not image_path.is_file():
                logger.warning("Skipping %s: image file not found", image_name)
                continue

            annotations = parse_auair_annotation(record)
            timestamp = record.get("time") or record.get("timestamp")

            samples.append(
                ImageSample(
                    image_id=Path(image_name).stem,
                    image_path=image_path,
                    annotations=annotations,
                    telemetry_timestamp=timestamp,
                )
            )

        logger.info(
            "AUAIRDataset[%s]: indexed %d samples from %s", self.split, len(samples), self.root
        )
        return samples
